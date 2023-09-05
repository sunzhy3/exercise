import os
import numpy as np

def get_output_sites(c, ks, stride, H_in, W_in, subm=False):
    out = []
    for i in range(c[0] - ks + 1, c[0] + 1):
        for j in range(c[1] - ks + 1, c[1] + 1):
            if i < 0 or i > H_in - ks or j < 0 or j > W_in - ks:
                continue
            out_i = i // stride
            out_j = j // stride
            out.append((out_i, out_j))
    return out

def get_offset(c, p, ks, stride):
    out_i, out_j = p
    src_i = out_i * stride
    src_j = out_j * stride
    offset_x = c[0] - (src_i + ks // 2)
    offset_y = c[1] - (src_j + ks // 2)
    return (offset_x, offset_y)

def spconv(input, in_coords, spatial_shape, kernel, stride=1, padding=0):
    """
    Args:
        input: (N_in, C_in)
        in_coords: (N_in, 2)
        kernel: (k_h, k_w, C_in, C_out)
    Return:
        output: (N_out, C_out)
    """
    N_in, C_in = input.shape
    H_in, W_in = spatial_shape
    k_h, k_w, C_in, C_out = kernel.shape
    ks = k_h

    hash_in = {}
    v_in = 0
    P_in = []
    for i in range(N_in):
        c = in_coords[i]
        hash_in[(c[0], c[1])] = v_in
        v_in += 1
        P_in.append((c[0], c[1]))
    
    hash_out = {}
    P_out = [[] for _ in range(N_in)]
    v_out = 0
    for i in range(N_in):
        c = in_coords[i]
        output_site = get_output_sites(c, ks, stride, H_in, W_in)
        for site in output_site:
            if site not in hash_out:
                hash_out[site] = v_out
                v_out += 1
        P_out[i] = output_site
    total_v_out = v_out
    
    # build rule book
    rulebook = []
    offset_counter = {}
    for i in range(N_in):
        c = in_coords[i]
        for p in P_out[i]:
            offset = get_offset(c, p, ks, stride)
            v_out = hash_out[p]
            if offset not in offset_counter:
                offset_counter[offset] = 0
            cnt = offset_counter[offset]
            rulebook.append([offset, cnt, i, v_out])
            offset_counter[offset] += 1

    output = np.zeros((total_v_out, C_out))
    for i in range(len(rulebook)):
        offset, count, v_in, v_out = rulebook[i]
        kc = (offset[0] + 1, offset[1] + 1)
        kw = kernel[kc[0], kc[1], :, :]  # (C_in, C_out)
        in_feat = input[v_in]  # (C_in)
        out_feat = np.dot(in_feat, kw)  # (C_out)
        # print(v_in, v_out, in_feat, kw)
        output[v_out, :] += out_feat
    
    return output

if __name__ == "__main__":
    input = np.random.rand(2, 3)
    spatial_shape = [5, 5]
    in_coords = np.array([[2, 1], [3, 2]])
    kernel = np.random.rand(3, 3, 3, 2)
    
    output = spconv(input, in_coords, spatial_shape, kernel)
    
    print(output)
    print(output.shape)


