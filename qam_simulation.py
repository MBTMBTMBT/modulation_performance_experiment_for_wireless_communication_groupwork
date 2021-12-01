import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import math
from tqdm import tqdm

import qam_tools


def simulate(M: int, N: int, batch_exp: int, num_workers: int):
    iter_times = 10 ** (N - batch_exp)
    N = batch_exp
    N0 = 1

    # get random bits
    bits = np.random.rand(int(math.log2(M)) * 10 * 10 ** N) < 0.5
    bits = np.transpose(np.reshape(bits, (int(math.log2(M)), bits.size//int(math.log2(M)))))

    # get symbols
    symbols = np.transpose(qam_tools.gray_mat_to_dec_vec(bits, int(math.log2(M))))
    symbol_mat = qam_tools.get_symbol_matrix(int(math.sqrt(M)))

    # start simulation
    Eb_over_N0_dB_values = np.linspace(-5, 15, 41)
    # Eb_over_N0_dB_values = [-5]
    # print(Eb_over_N0_dB_values)
    bit_error_rates = Eb_over_N0_dB_values.copy()

    # create waitbar
    pbar = tqdm(desc="Simulating", total=len(Eb_over_N0_dB_values))
    fig_count = 1

    # create pool
    # parallel_manager = multiprocessing.Manager()
    parallel_pool = multiprocessing.Pool(num_workers)

    for count, Eb_over_N0_dB in enumerate(Eb_over_N0_dB_values):
        pbar.update()

        Eb_over_N0 = 10 ** (Eb_over_N0_dB/10)
        Es_over_N0 = Eb_over_N0 * 4
        d_square = (1 / qam_tools.compute_Es_over_d_square(symbol_mat)) * Es_over_N0 * N0
        d = math.sqrt(d_square)

        mapping_list = []
        # run in parallel
        para_dict = {
            'symbols': symbols,
            'bits': bits,
            'symbol_mat': symbol_mat,
            'd': d,
            'N0': N0,
            'plot': False,
            'fig_num': 0
        }
        for i in range(iter_times):
            if i == 0 and Eb_over_N0_dB == 5:
                para_dict['plot'] = True
                para_dict['fig_num'] = fig_count
                fig_count += 1
            mapping_list.append(para_dict)
            # run_parallel(para_dict)
        compute_rst = parallel_pool.map(run_parallel, mapping_list)
        count_err = 0
        count_bits = 0
        for each_rst in compute_rst:
            count_err += each_rst[0]
            count_bits += each_rst[1]

        bit_error_rates[count] = count_err / count_bits
    
    '''
    plt.figure(fig_count)
    fig_count += 1
    plt.semilogy(Eb_over_N0_dB_values, bit_error_rates)
    plt.title('BER vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.show()
    '''

    return Eb_over_N0_dB_values, bit_error_rates


def run_parallel(para_dict: dict):
    symbols = para_dict['symbols']
    bits = para_dict['bits']
    symbols_mat = para_dict['symbol_mat']
    d = para_dict['d']
    N0 = para_dict['N0']
    plot = para_dict['plot']
    fig_num = para_dict['fig_num']
    rst = run_para(symbols, bits, symbols_mat, d, N0, plot, fig_num)
    return rst


def run_para(symbols: np.ndarray, bits: np.ndarray, symbol_mat: np.ndarray, 
    d: float, N0: float, plot: bool, fig_num: int):
    symbols_x = qam_tools.compute_symbol_xaxis(symbols, d, symbol_mat)
    symbols_y = qam_tools.compute_symbol_yaxis(symbols, d, symbol_mat)

    noise_x = np.random.randn(symbols.shape[0], symbols.shape[1]) * math.sqrt(N0 / 2)
    noise_y = np.random.randn(symbols.shape[0], symbols.shape[1]) * math.sqrt(N0 / 2)

    symbols_x += noise_x
    symbols_y += noise_y

    decided = qam_tools.decide(symbols_x, symbols_y, d, symbol_mat)
    decided_bits = qam_tools.dec_vec_to_gray_mat(decided, int(math.log2(symbol_mat.size)))

    # errs = np.bitwise_xor(np.squeeze(decided_bits).astype(np.int8), np.squeeze(bits.astype(np.int8)))
    err_num = np.sum(decided_bits.astype(np.int8) != bits.astype(np.int8))
    bit_num = bits.shape[0] * bits.shape[1]

    if plot:
        # qam_tools.plot_scatter(symbols, symbols_x, symbols_y, 5000, fig_num)
        pass

    return err_num, bit_num


if __name__ == '__main__':
    _, bit_error_rates_4 = simulate(4, 6, 4, 16)
    _, bit_error_rates_16 = simulate(16, 6, 4, 16)
    Eb_over_N0_dB_values, bit_error_rates_64 = simulate(64, 6, 4, 16)
    plt.figure(0)
    plt.semilogy(Eb_over_N0_dB_values, bit_error_rates_4, label='Offset QPSK')
    plt.semilogy(Eb_over_N0_dB_values, bit_error_rates_16, label='16-QAM')
    plt.semilogy(Eb_over_N0_dB_values, bit_error_rates_64, label='64-QAM')
    plt.title('BER vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.legend()
    plt.show()
