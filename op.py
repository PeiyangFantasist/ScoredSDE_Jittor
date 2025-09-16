import jittor as jt
from jittor import Function
import jittor.nn as nn
import numpy as np
import os

jt.flags.use_cuda = 1  # 启用CUDA


def upfirdn2d(
    input: jt.Var,    # 输入张量，形状：(major_dim, in_h, in_w, minor_dim)
    kernel: jt.Var,   # 滤波核张量，形状：(kernel_h, kernel_w)
    up_x: int,        # x方向上采样倍数
    up_y: int,        # y方向上采样倍数
    down_x: int,      # x方向下采样倍数
    down_y: int,      # y方向下采样倍数
    pad_x0: int,      # x方向左填充
    pad_x1: int,      # x方向右填充
    pad_y0: int,      # y方向上填充
    pad_y1: int       # y方向下填充
) -> jt.Var:
    """
    Upfirdn2d API to CUDA
    """
    # -------------------------- 1. 计算输出张量形状（与原核函数逻辑一致）--------------------------
    major_dim = input.shape[0]  # 对应 p.major_dim（如批量维度）
    in_h = input.shape[1]       # 对应 p.in_h（输入高度）
    in_w = input.shape[2]       # 对应 p.in_w（输入宽度）
    minor_dim = input.shape[3]  # 对应 p.minor_dim（如通道维度）
    kernel_h = kernel.shape[0]  # 对应 p.kernel_h（滤波核高度）
    kernel_w = kernel.shape[1]  # 对应 p.kernel_w（滤波核宽度）

    # 计算输出高度/宽度（原核函数 p.out_h/p.out_w 的计算公式）
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    # 定义线程配置参数（与原核函数一致：分块处理大任务）
    loop_major = (major_dim - 1) // 16384 + 1  # 对应 p.loop_major
    loop_x = 4                                 # 对应 p.loop_x
    block_size_x = 4                           # 对应原 blockDim.x
    block_size_y = 32                          # 对应原 blockDim.y


    # -------------------------- 2. 调用 jt.code 实现 CUDA 逻辑--------------------------
    return jt.code(
        shape=(major_dim, out_h, out_w, minor_dim),  # 输出张量形状
        dtype=input.dtype,                            # 输出数据类型与输入一致
        # inputs = [@in0, @in1, @in2, @in3, @in4, @in5,   @in6,   @in7,    @in8,   @in9]
        inputs=[input, kernel, jt.array([up_x], dtype=jt.int32), 
        jt.array([up_y], dtype=jt.int32),
        jt.array([down_x], dtype=jt.int32),
        jt.array([down_y], dtype=jt.int32),
        jt.array([pad_x0], dtype=jt.int32),
        jt.array([pad_x1], dtype=jt.int32),
        jt.array([pad_y0], dtype=jt.int32),
        jt.array([pad_y1], dtype=jt.int32)],
        # CUDA API
        cuda_header='''
        #include <cuda.h>
        #include <cuda_runtime.h>
        ''',
        cuda_src='''
            __global__ static void jt_upfirdn2d_kernel(@ARGS_DEF) {
                @PRECALC
                @alias(input, in0);
                @alias(kernel, in1);
                @alias(up_x, in2);
                @alias(up_y, in3);
                @alias(down_x, in4);
                @alias(down_y, in5);
                @alias(pad_x0, in6);
                @alias(pad_x1, in7);
                @alias(pad_y0, in8);
                @alias(pad_y1, in9);

                const int major_dim = input_shape0;
                const int in_h = input_shape1;
                const int in_w = input_shape2;
                const int minor_dim = input_shape3;
                const int kernel_h = kernel_shape0;
                const int kernel_w = kernel_shape1;
                int out_h = (in_h * @up_y(0) + @pad_y0(0) + @pad_y1(0) - kernel_shape0 + @down_y(0)) / @down_y(0);
                int out_w = (in_w * @up_x(0) + @pad_x0(0) + @pad_x1(0) - kernel_shape1 + @down_x(0)) / @down_x(0);
                const int res_int = (major_dim - 1) / 16384;
                const int loop_major = res_int + 1;
                const int loop_x = 4;

                int minor_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 对应原 minor_idx
                int out_y = minor_idx / minor_dim;                     // 对应原 out_y
                minor_idx -= out_y * minor_dim;                        // 修正 minor_idx

                int out_x_base = blockIdx.y * loop_x * blockDim.y + threadIdx.y;  // 对应原 out_x_base
                int major_idx_base = blockIdx.z * loop_major;                    // 对应原 major_idx_base

                // 边界检查：超出输出范围的线程直接退出（避免无效计算）
                if (out_x_base >= out_w || out_y >= out_h || major_idx_base >= major_dim) {
                    return;
                }

                int mid_y = out_y * @down_y(0) + @up_y(0) - 1 - @pad_y0(0);  // 对应原 mid_y
                
                int in_y = min(max((mid_y >= 0 ? mid_y / @up_y(0) : (mid_y - @up_y(0) + 1) / @up_y(0)), 0), in_h);  // 对应原 floor_div
                
                int h = min(max((mid_y + kernel_h >= 0 ? (mid_y + kernel_h) / @up_y(0) : ((mid_y + kernel_h) - @up_y(0) + 1) / @up_y(0)), 0), in_h) - in_y;  // 对应原 h
                
                int kernel_y = mid_y + kernel_h - (in_y + 1) * @up_y(0);  // 对应原 kernel_y

                // 遍历批量维度（major_dim）：分 loop_major 次处理
                for (int loop_major_cnt = 0, major_idx = major_idx_base;
                     loop_major_cnt < loop_major && major_idx < major_dim;
                     loop_major_cnt++, major_idx++) {
                    
                    // 遍历 x 方向：分 loop_x 次处理（每次处理 blockDim.y 个 out_x）
                    for (int loop_x_cnt = 0, out_x = out_x_base;
                         loop_x_cnt < loop_x && out_x < out_w;
                         loop_x_cnt++, out_x += blockDim.y) {

                        int mid_x = out_x * @down_x(0) + @up_x(0) - 1 - @pad_x0(0);  // 对应原 mid_x
                        int in_x = min(max((mid_x >= 0 ? mid_x / @up_x(0) : (mid_x - @up_x(0) + 1) / @up_x(0)), 0), in_w);  // 对应原 floor_div
                        int w = min(max((mid_x + kernel_w >= 0 ? (mid_x + kernel_w) / @up_x(0) : ((mid_x + kernel_w) - @up_x(0) + 1) / @up_x(0)), 0), in_w) - in_x;  // 对应原 w
                        int kernel_x = mid_x + kernel_w - (in_x + 1) * @up_x(0);  // 对应原 kernel_x

                        float val = 0.0f;  // 存储当前输出像素的累加值

                        // 遍历滤波核 y 方向覆盖的输入行
                        for (int y = 0; y < h; y++) {
                            // 遍历滤波核 x 方向覆盖的输入列
                            for (int x = 0; x < w; x++) {
                                // 1. 读取输入张量对应位置的值（@in0 访问 input，索引：(major_idx, in_y+y, in_x+x, minor_idx)）
                                float input_val = @in0(major_idx, in_y + y, in_x + x, minor_idx);
                                // 2. 读取滤波核对应位置的权重（@in1 访问 kernel，索引：(kernel_y+y, kernel_x+x)）
                                float kernel_val = @in1(kernel_y + y, kernel_x + x);
                                // 3. 累加：输入值 × 核权重
                                val += input_val * kernel_val;
                            }
                        }
                        @out(major_idx, out_y, out_x, minor_idx) = val;
                    }
                }
            }
           jt_upfirdn2d_kernel<<<32, 32>>>(@ARGS);
        '''
    )

class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, jt.misc.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    '''Interface Function'''
    if input.device.type == "cpu":
        out = upfirdn2d_native(
            input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
        )

    else:
        out = UpFirDn2d.apply(
            input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
        )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = nn.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = nn.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = jt.misc.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = nn.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)