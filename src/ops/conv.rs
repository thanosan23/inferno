use crate::backend;
use crate::ops::layout::{swap_leading_axes, transpose_matrix};
use crate::tensor::Tensor;

struct Conv2dGeometry {
    batch: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    out_height: usize,
    out_width: usize,
}

impl Conv2dGeometry {
    fn new(input_shape: &[usize], weight_shape: &[usize], stride: usize, padding: usize) -> Conv2dGeometry {
        assert_eq!(input_shape.len(), 4, "conv2d: input must be [batch, in_channels, height, width], got {:?}", input_shape);
        assert_eq!(
            weight_shape.len(),
            4,
            "conv2d: weight must be [out_channels, in_channels, kernel_height, kernel_width], got {:?}",
            weight_shape
        );
        let [batch, in_channels, in_height, in_width] = <[usize; 4]>::try_from(input_shape).unwrap();
        let [out_channels, weight_in_channels, kernel_height, kernel_width] = <[usize; 4]>::try_from(weight_shape).unwrap();
        assert_eq!(in_channels, weight_in_channels, "conv2d: input has {in_channels} channels but weight expects {weight_in_channels}");

        let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;

        Conv2dGeometry {
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
            stride,
            padding,
            out_height,
            out_width,
        }
    }

    fn patch_size(&self) -> usize {
        self.in_channels * self.kernel_height * self.kernel_width
    }

    fn output_positions(&self) -> usize {
        self.batch * self.out_height * self.out_width
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.batch, self.out_channels, self.out_height, self.out_width]
    }

    fn input_len(&self) -> usize {
        self.batch * self.in_channels * self.in_height * self.in_width
    }

    fn for_each_patch_element(&self, mut visit: impl FnMut(usize, usize, usize)) {
        for channel in 0..self.in_channels {
            for kernel_row in 0..self.kernel_height {
                for kernel_col in 0..self.kernel_width {
                    let patch_row = (channel * self.kernel_height + kernel_row) * self.kernel_width + kernel_col;
                    for batch in 0..self.batch {
                        for out_row in 0..self.out_height {
                            let padded_row = out_row * self.stride + kernel_row;
                            if padded_row < self.padding || padded_row - self.padding >= self.in_height {
                                continue;
                            }
                            let in_row = padded_row - self.padding;

                            for out_col in 0..self.out_width {
                                let padded_col = out_col * self.stride + kernel_col;
                                if padded_col < self.padding || padded_col - self.padding >= self.in_width {
                                    continue;
                                }
                                let in_col = padded_col - self.padding;

                                let output_position = (batch * self.out_height + out_row) * self.out_width + out_col;
                                let input_index = ((batch * self.in_channels + channel) * self.in_height + in_row) * self.in_width + in_col;
                                visit(patch_row, output_position, input_index);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn im2col(input: &[f32], geometry: &Conv2dGeometry) -> Vec<f32> {
    let output_positions = geometry.output_positions();
    let mut columns = vec![0f32; geometry.patch_size() * output_positions];
    geometry.for_each_patch_element(|patch_row, output_position, input_index| {
        columns[patch_row * output_positions + output_position] = input[input_index];
    });
    columns
}

fn col2im(column_grad: &[f32], geometry: &Conv2dGeometry) -> Vec<f32> {
    let output_positions = geometry.output_positions();
    let mut input_grad = vec![0f32; geometry.input_len()];
    geometry.for_each_patch_element(|patch_row, output_position, input_index| {
        input_grad[input_index] += column_grad[patch_row * output_positions + output_position];
    });
    input_grad
}

fn gemm_output_to_nchw(gemm_output: &[f32], geometry: &Conv2dGeometry) -> Vec<f32> {
    swap_leading_axes(gemm_output, geometry.out_channels, geometry.batch, geometry.out_height * geometry.out_width)
}

fn nchw_grad_to_gemm_layout(output_grad: &[f32], geometry: &Conv2dGeometry) -> Vec<f32> {
    swap_leading_axes(output_grad, geometry.batch, geometry.out_channels, geometry.out_height * geometry.out_width)
}

struct Pool2dGeometry {
    batch: usize,
    channels: usize,
    in_height: usize,
    in_width: usize,
    kernel: usize,
    stride: usize,
    out_height: usize,
    out_width: usize,
}

impl Pool2dGeometry {
    fn new(input_shape: &[usize], kernel: usize, stride: usize) -> Pool2dGeometry {
        assert_eq!(input_shape.len(), 4, "expected [batch, channels, height, width], got {:?}", input_shape);
        let [batch, channels, in_height, in_width] = <[usize; 4]>::try_from(input_shape).unwrap();
        let out_height = (in_height - kernel) / stride + 1;
        let out_width = (in_width - kernel) / stride + 1;
        Pool2dGeometry { batch, channels, in_height, in_width, kernel, stride, out_height, out_width }
    }

    fn planes(&self) -> usize {
        self.batch * self.channels
    }

    fn output_len(&self) -> usize {
        self.planes() * self.out_height * self.out_width
    }

    fn output_shape(&self) -> Vec<usize> {
        vec![self.batch, self.channels, self.out_height, self.out_width]
    }

    fn window(&self, plane: usize, out_row: usize, out_col: usize) -> impl Iterator<Item = usize> + '_ {
        let plane_offset = plane * self.in_height * self.in_width;
        (0..self.kernel).flat_map(move |kernel_row| {
            (0..self.kernel).map(move |kernel_col| {
                let in_row = out_row * self.stride + kernel_row;
                let in_col = out_col * self.stride + kernel_col;
                plane_offset + in_row * self.in_width + in_col
            })
        })
    }

    fn for_each_output_position(&self, mut visit: impl FnMut(usize, usize, usize)) {
        for plane in 0..self.planes() {
            for out_row in 0..self.out_height {
                for out_col in 0..self.out_width {
                    visit(plane, out_row, out_col);
                }
            }
        }
    }
}

impl Tensor {
    pub fn conv2d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        let geometry = Conv2dGeometry::new(&self.shape(), &weight.shape(), stride, padding);

        let input_data = self.data();
        let weight_data = weight.data();
        let columns = im2col(&input_data, &geometry);
        let gemm_output = backend::matmul(&weight_data, &columns, geometry.out_channels, geometry.patch_size(), geometry.output_positions());
        let output = gemm_output_to_nchw(&gemm_output, &geometry);

        Tensor::make(output, geometry.output_shape(), vec![self.clone(), weight.clone()], move |output_grad| {
            let gemm_output_grad = nchw_grad_to_gemm_layout(output_grad, &geometry);

            let weight_transposed = transpose_matrix(&weight_data, geometry.out_channels, geometry.patch_size());
            let column_grad = backend::matmul(&weight_transposed, &gemm_output_grad, geometry.patch_size(), geometry.out_channels, geometry.output_positions());
            let input_grad = col2im(&column_grad, &geometry);

            let columns_transposed = transpose_matrix(&columns, geometry.patch_size(), geometry.output_positions());
            let weight_grad = backend::matmul(&gemm_output_grad, &columns_transposed, geometry.out_channels, geometry.output_positions(), geometry.patch_size());

            vec![input_grad, weight_grad]
        })
    }

    pub fn conv1d(&self, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        assert_eq!(input_shape.len(), 3, "conv1d: input must be [batch, in_channels, length], got {:?}", input_shape);
        assert_eq!(weight_shape.len(), 3, "conv1d: weight must be [out_channels, in_channels, kernel_length], got {:?}", weight_shape);

        let input_2d = self.reshape(&[input_shape[0], input_shape[1], 1, input_shape[2]]);
        let weight_2d = weight.reshape(&[weight_shape[0], weight_shape[1], 1, weight_shape[2]]);
        let output_2d = input_2d.conv2d(&weight_2d, stride, padding);

        let output_shape = output_2d.shape();
        output_2d.reshape(&[output_shape[0], output_shape[1], output_shape[3]])
    }

    pub fn max_pool2d(&self, kernel: usize, stride: usize) -> Tensor {
        let geometry = Pool2dGeometry::new(&self.shape(), kernel, stride);
        let input = self.data();

        let mut output = Vec::with_capacity(geometry.output_len());
        let mut argmax = Vec::with_capacity(geometry.output_len());
        geometry.for_each_output_position(|plane, out_row, out_col| {
            let (best_index, best_value) = geometry
                .window(plane, out_row, out_col)
                .map(|index| (index, input[index]))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            output.push(best_value);
            argmax.push(best_index);
        });

        let input_len = input.len();
        Tensor::make(output, geometry.output_shape(), vec![self.clone()], move |output_grad| {
            let mut input_grad = vec![0f32; input_len];
            for (position, &index) in argmax.iter().enumerate() {
                input_grad[index] += output_grad[position];
            }
            vec![input_grad]
        })
    }

    pub fn avg_pool2d(&self, kernel: usize, stride: usize) -> Tensor {
        let geometry = Pool2dGeometry::new(&self.shape(), kernel, stride);
        let input = self.data();
        let window_size = (kernel * kernel) as f32;

        let mut output = Vec::with_capacity(geometry.output_len());
        geometry.for_each_output_position(|plane, out_row, out_col| {
            let sum: f32 = geometry.window(plane, out_row, out_col).map(|index| input[index]).sum();
            output.push(sum / window_size);
        });

        let output_shape = geometry.output_shape();
        let input_len = input.len();
        Tensor::make(output, output_shape, vec![self.clone()], move |output_grad| {
            let mut input_grad = vec![0f32; input_len];
            let mut position = 0usize;
            geometry.for_each_output_position(|plane, out_row, out_col| {
                let grad = output_grad[position] / window_size;
                for index in geometry.window(plane, out_row, out_col) {
                    input_grad[index] += grad;
                }
                position += 1;
            });
            vec![input_grad]
        })
    }

    pub fn embedding(&self, indices: &[usize]) -> Tensor {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "embedding: weight must be [num_embeddings, dim], got {:?}", shape);
        let (num_embeddings, dim) = (shape[0], shape[1]);
        let table = self.data();

        let mut output = Vec::with_capacity(indices.len() * dim);
        for &index in indices {
            assert!(index < num_embeddings, "embedding index {index} out of range ({num_embeddings} rows)");
            output.extend_from_slice(&table[index * dim..index * dim + dim]);
        }

        let indices = indices.to_vec();
        let table_len = table.len();
        Tensor::make(output, vec![indices.len(), dim], vec![self.clone()], move |output_grad| {
            let mut table_grad = vec![0f32; table_len];
            for (row, &index) in indices.iter().enumerate() {
                for d in 0..dim {
                    table_grad[index * dim + d] += output_grad[row * dim + d];
                }
            }
            vec![table_grad]
        })
    }
}
