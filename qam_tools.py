import numpy as np
import matplotlib.pyplot as plt
import math


def gray_code(n: int):
    if n == 1: return ['0','1']
    return ['0'+i for i in gray_code(n-1)] + ['1'+i for i in gray_code(n-1)[::-1]]


def swap_gray_code(gray_code_strs: list):
    length = len(gray_code_strs)
    side_length = int(math.sqrt(length))
    for i in range(side_length):
        if i % 2 != 0:
            for j in range(side_length//2):
                gray_code_strs[side_length*i+j], gray_code_strs[side_length*(i+1)-1-j] \
                    = gray_code_strs[side_length*(i+1)-1-j], gray_code_strs[side_length*i+j]


def get_dec_to_gray_code_mapping(n: int) -> list:
    gray_code_strs = gray_code(n)
    swap_gray_code(gray_code_strs)
    rst = []
    for i in range(len(gray_code_strs)):
        rst.append(int(gray_code_strs[i], 2))
    return rst


def get_gray_code_to_dec_mapping(n: int) -> list:
    gray_code_strs = gray_code(n)
    swap_gray_code(gray_code_strs)
    rst = [i for i in range(len(gray_code_strs))]
    for i in range(len(gray_code_strs)):
        idx = int(gray_code_strs[i], 2)
        rst[idx] = i
    return rst


def dec_vec_to_bin_mat(dec_vec: np.ndarray, bin_bits: int) -> np.ndarray:
    dec_vec = dec_vec.copy()
    bin_mat = np.zeros((len(dec_vec), bin_bits))
    divider = 2 ** (bin_bits - 1)
    for i in range(bin_bits):
        bin_mat[:, i] = dec_vec >= divider
        dec_vec = dec_vec - bin_mat[:, i] * divider
        divider /= 2
    return bin_mat


def bin_mat_to_dec_vec(bin_mat: np.ndarray) -> np.ndarray:
    bin_bits = bin_mat.shape[1]
    dec_vec = np.zeros((bin_mat.shape[0], 1))
    multiplexer = 2 ** (bin_bits - 1)
    for i in range(bin_bits):
        dec_vec = dec_vec + bin_mat[:, i, np.newaxis] * multiplexer
        multiplexer /= 2
    return dec_vec


def dec_vec_to_gray_mat(dec_vec: np.ndarray, num_bits: int) -> np.ndarray:
    dec_vec = dec_vec.astype(np.int64)
    map = np.array(get_dec_to_gray_code_mapping(num_bits))
    gray_dec_vec = map[dec_vec]
    gray_mat = dec_vec_to_bin_mat(gray_dec_vec, num_bits)
    return gray_mat


def gray_mat_to_dec_vec(gray_mat: np.ndarray, num_bits: int) -> np.ndarray:
    map = np.array(get_gray_code_to_dec_mapping(num_bits))
    gray_dec_vec = bin_mat_to_dec_vec(gray_mat).astype(np.int64)
    dec_vec = map[gray_dec_vec]
    return dec_vec


def get_symbol_matrix(n: int) -> np.ndarray:
    rst = np.array([i for i in range(n**2)])
    rst = np.reshape(rst, (n, n))
    return rst


def compute_Es_over_d_square(symbol_matrix: np.ndarray):
    side_length = symbol_matrix.shape[0]
    mid_col = (side_length - 1) / 2
    count = 0
    energy = 0
    for i in range(side_length):
        for j in range(side_length):
            symbol_energy = (i-mid_col)**2 + (j-mid_col)**2
            count += 1
            energy += symbol_energy
    return energy / count


def compute_symbol_xaxis(symbols: np.ndarray, d: float, symbol_mat: np.ndarray) -> np.ndarray:
    # expected to be a square matrix
    assert symbol_mat.shape[0] == symbol_mat.shape[1]
    side_length = symbol_mat.shape[0]
    symbols_x = np.zeros(symbols.shape, dtype=np.float64)
    mid_col = (side_length - 1) / 2
    for col in range(side_length):
        dis = col - mid_col
        distance = dis * d
        check = symbol_mat[:, col]
        symbols_x += np.dot(np.isin(symbols, symbol_mat[:, col]), distance)
    return symbols_x


def compute_symbol_yaxis(symbols: np.ndarray, d: float, symbol_mat: np.ndarray) -> np.ndarray:
    # expected to be a square matrix
    assert symbol_mat.shape[0] == symbol_mat.shape[1]
    side_length = symbol_mat.shape[0]
    symbols_y = np.zeros(symbols.shape, dtype=np.float64)
    mid_row = (side_length - 1) / 2
    for row in range(side_length):
        dis = mid_row - row
        distance = dis * d
        check = symbol_mat[row, :]
        symbols_y += np.dot(np.isin(symbols, symbol_mat[row, :]), distance)
    return symbols_y


def decide(symbols_x: np.ndarray, symbols_y: np.ndarray, 
            d: float, symbol_mat: np.ndarray) -> np.ndarray:
    assert symbol_mat.shape[0] == symbol_mat.shape[1]
    side_length = symbol_mat.shape[0]
    distances = []
    mid_col = (side_length - 1) / 2
    for col in range(side_length):
        dis = col - mid_col
        distances.append(dis * d)
    dec_bound_x = []
    for i in range(len(distances) - 1):
        dec_bound_x.append(distances[i] + d/2)
    dec_bound_y = list(reversed(dec_bound_x))

    r_hori = [symbols_x < dec_bound_x[0]]
    r_verti = [symbols_y >= dec_bound_y[0]]
    for i in range(side_length - 2):
        r_hori.append((symbols_x <= dec_bound_x[i+1]) & (symbols_x > dec_bound_x[i]))
        r_verti.append((symbols_y >= dec_bound_y[i+1]) & (symbols_y < dec_bound_y[i]))
    r_hori.append(symbols_x > dec_bound_x[-1])
    r_verti.append(symbols_y < dec_bound_y[-1])

    d_mat = np.zeros((np.max(symbol_mat)+1, symbols_x.shape[0]*symbols_x.shape[1]), dtype=bool)
    idx = 0
    for i in range(side_length):
        for j in range(side_length):
            d_mat[idx, :] = r_hori[j] & r_verti[i]
            idx += 1
    
    d_symbols = d_mat[0, :].astype(np.int32) * 0
    for i in range(np.max(symbol_mat) + 1):
        d_symbols += np.dot(d_mat[i, :].astype(np.int32), i)
    
    return d_symbols


def plot_scatter(symbols: np.ndarray, symbols_x: np.ndarray, symbols_y: np.ndarray, 
                    plot_size: int, fig_num: int) -> None:
    plotted_symbols = np.squeeze(symbols)[0:plot_size]
    plot_rx = np.squeeze(symbols_x)[0:plot_size]
    plot_ry = np.squeeze(symbols_y)[0:plot_size]
    plotted_x = []
    plotted_y = []
    for i in range(np.max(symbols) + 1):
        symbs_x = []
        symbs_y = []
        for j in range(plotted_symbols.size):
            if plotted_symbols[j] == i:
                symbs_x.append(plot_rx[j])
                symbs_y.append(plot_ry[j])
        plotted_x.append(symbols_x)
        plotted_y.append(symbols_y)
    plt.figure(fig_num)
    for i in range(np.max(symbols) + 1):
        hold = not i == np.max(symbols)
        label = "symbol %d" % (i + 1)
        plt.scatter(plotted_x[i], plotted_y[i], label=label)
    plt.title("Received Symbols")
    plt.xlabel('\phi1')
    plt.ylabel('\phi2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    get_dec_to_gray_code_mapping(4)
