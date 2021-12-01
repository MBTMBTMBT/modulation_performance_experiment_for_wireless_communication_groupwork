import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import math
from tqdm import tqdm

import qam_tools


def simulate_retransmission(M: int, test_frames: int, frame_size: int, num_workers: int):
    N0 = 1

    assert frame_size % int(math.log2(M)) == 0
    symbols_per_frame = frame_size // int(math.log2(M))

    # get random bits
    bits = np.random.rand(test_frames * frame_size) < 0.5
    bits = np.transpose(np.reshape(bits, (int(math.log2(M)), bits.size//int(math.log2(M)))))

    # get symbols
    symbols = np.transpose(qam_tools.gray_mat_to_dec_vec(bits, int(math.log2(M))))
    symbol_mat = qam_tools.get_symbol_matrix(int(math.sqrt(M)))

    # start simulation
    # Eb_over_N0_dB_values = np.linspace(-5, 15, 41)
    Eb_over_N0_dB_values = np.linspace(0, 25, 51)
    # Eb_over_N0_dB_values = np.linspace(-10, 20, 61)
    # Eb_over_N0_dB_values = [15]
    # print(Eb_over_N0_dB_values)
    retransmissions = Eb_over_N0_dB_values.copy()

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
        for i in range(test_frames):
            para_dict = {
                'symbols': symbols[0, symbols_per_frame*i:symbols_per_frame*(i+1)][np.newaxis, :],
                'bits': bits[frame_size//int(math.log2(M))*i:frame_size//int(math.log2(M))*(i+1), :],
                'symbol_mat': symbol_mat,
                'd': d,
                'N0': N0,
                'plot': False,
                'fig_num': 0
            }
            if i == 0 and Eb_over_N0_dB == 5:
                para_dict['plot'] = True
                para_dict['fig_num'] = fig_count
                fig_count += 1
            mapping_list.append(para_dict)
            # run_parallel(para_dict)
        rsts = parallel_pool.map(run_parallel, mapping_list)
        total_retransmissions = 0
        for each_rst in rsts:
            total_retransmissions += each_rst

        retransmissions[count] = total_retransmissions
    
    '''
    plt.figure(fig_count)
    fig_count += 1
    plt.semilogy(Eb_over_N0_dB_values, bit_error_rates)
    plt.title('BER vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.show()
    '''

    return Eb_over_N0_dB_values, retransmissions


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

    count_retransmit = 0
    while True:
        noise_x = np.random.randn(symbols.shape[0], symbols.shape[1]) * math.sqrt(N0 / 2)
        noise_y = np.random.randn(symbols.shape[0], symbols.shape[1]) * math.sqrt(N0 / 2)

        symbols_x += noise_x
        symbols_y += noise_y

        decided = qam_tools.decide(symbols_x, symbols_y, d, symbol_mat)
        decided_bits = qam_tools.dec_vec_to_gray_mat(decided, int(math.log2(symbol_mat.size)))

        # errs = np.bitwise_xor(np.squeeze(decided_bits).astype(np.int8), np.squeeze(bits.astype(np.int8)))
        err_num = np.sum(decided_bits.astype(np.int8) != bits.astype(np.int8))
        # print(err_num)
        if err_num == 0 or count_retransmit >= 250:
            if count_retransmit >= 250:
                count_retransmit = 10000
                pass
            break
        count_retransmit += 1

    if plot:
        # qam_tools.plot_scatter(symbols, symbols_x, symbols_y, 5000, fig_num)
        pass

    return count_retransmit


if __name__ == '__main__':
    _, retransmissions_4 = simulate_retransmission(4, 256, 12, 16)
    _, retransmissions_16 = simulate_retransmission(16, 256, 12, 16)
    Eb_over_N0_dB_values, retransmissions_64 = simulate_retransmission(64, 256, 12, 16)
    plt.figure(0)
    plt.plot(Eb_over_N0_dB_values, retransmissions_4, label='Offset QPSK')
    plt.plot(Eb_over_N0_dB_values, retransmissions_16, label='16-QAM')
    plt.plot(Eb_over_N0_dB_values, retransmissions_64, label='64-QAM')
    plt.title('retransmissions vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('retransmissions')
    plt.legend()
    plt.figure(1)
    plt.semilogy(Eb_over_N0_dB_values, (retransmissions_4+1)*(12/math.log2(4)), label='Offset QPSK')
    plt.semilogy(Eb_over_N0_dB_values, (retransmissions_16+1)*(12/math.log2(16)), label='16-QAM')
    plt.semilogy(Eb_over_N0_dB_values, (retransmissions_64+1)*(12/math.log2(64)), label='64-QAM')
    plt.title('transmitted symbols vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('transmitted symbols')
    plt.legend()
    plt.figure(2)
    plt.semilogy(Eb_over_N0_dB_values, 1 / ((retransmissions_4+1)*(12/math.log2(4))), label='Offset QPSK')
    plt.semilogy(Eb_over_N0_dB_values, 1 / ((retransmissions_16+1)*(12/math.log2(16))), label='16-QAM')
    plt.semilogy(Eb_over_N0_dB_values, 1 / ((retransmissions_64+1)*(12/math.log2(64))), label='64-QAM')
    plt.title('information rate vs Eb/N0')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('information rate')
    plt.legend()
    plt.show()
