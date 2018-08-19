import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(0)#乱数seed固定

# 簡単な信号の作成(正弦波 + ノイズ)
N = 128 # サンプル数
dt = 0.01 # サンプリング周期(sec)
freq = 4 # 周波数(10Hz)
amp = 1 # 振幅

t = np.arange(0, N*dt, dt) # 時間軸
f = amp * np.sin(2 * np.pi*freq * t) + np.random.randn(N)*0.3 # 信号
f[:10]
plt.xlabel('time(sec)',fontsize=14)
plt.ylabel('signal',fontsize=14)
plt.plot(t,f)
# 高速フーリエ変換
F = np.fft.fft(f)
F[:10]
# FFTの複素数の結果を絶対値に変換
F_abs = np.abs(F)
# 振幅をもとの信号に揃える
F_abs_amp = F_abs / N * 2 # 交流成分はデータ数で割って2倍
F_abs_amp[0] = F_abs_amp[0] / 2 # 直流成分は2倍不要

fq = np.linspace(0, 1.0/dt, N) # 周波数軸 linespace(開始、終了、分割数)
# グラフ表示
plt.xlabel('frequency(Hz)', fontsize=14)
plt.ylabel('amplitude', fontsize=14)
plt.plot(fq,F_abs_amp)

# そのまま普通にIFFTで逆変換した場合
F_ifft = np.fft.ifft(F)
F_ifft_real = F_ifft.real
plt.plot(t, F_ifft_real, c = "g")

F2 = np.copy(F)
F2[:10]

# 周波数でフィルタリング処理
fc = 10 # カットオフ
F2[(fq > fc)] = 0
# カットオフを超える周波数のデータをゼロにする(ノイズ除去)

# フィルタリング処理したFFT結果の確認
# FFTの複素数結果を絶対値に変換
F2_abs = np.abs(F2)
# 振幅をもとの信号に揃える
F2_abs_amp = F2_abs / N * 2 # 交流成分はデータ数で割って2倍
F2_abs_amp[0] = F2_abs_amp[0] /2 # 直流成分は2倍不要

# グラフ表示
plt.xlabel('frequency(Hz)',fontsize=14)
plt.ylabel('amplitude',fontsize=14)
plt.plot(fq,F2_abs_amp,c='r')
# 周波数でフィルタリング(ノイズ除去) => IFFT
F2_ifft = np.fft.ifft(F2) #IFFT

F2_ifft_real = F2_ifft.real * 2 # 実数部の取得、振幅を元のスケールに戻す

plt.plot(t, f, label='original')
plt.plot(t, F2_ifft_real, c="r", linewidth=4, alpha=0.7, label='filtered')
plt.legend(loc='best')
plt.xlabel('time(sec)', fontsize=14)
plt.ylabel('singnal', fontsize=14)

plt.xlabel('frequency(Hz)',fontsize=14)
plt.ylabel('amplitude',fontsize=14)
plt.hlines(y=[0.2],xmin=0,xmax=100,colors='r',linestyles='dashed')
plt.plot(fq,F_abs_amp)

F3 = np.copy(F) # FFT結果をコピー
# 振幅強度でフィルタリング処理
F3 = np.copy(F)# FFT結果をコピー
ac = 0.2 # 振幅強度の閾値
F3[(F_abs_amp < ac)] = 0 # 振幅が閾値未満はゼロにする(ノイズ除去)

# 振幅でフィルタリング処理をした結果の確認
# FFTの複素数結果を絶対値に変換
F3_abs = np.abs(F3)
# 振幅を元の信号に揃える
F3_abs_amp = F3_abs / N*2 # 交流成分はデータ数で割って2倍
F3_abs_amp[0] = F3_abs_amp[0] / 2 # 直流成分は2倍不要

# グラフ表示
plt.xlabel('frequency(Hz)',fontsize=14)
plt.ylabel('amplitude',fontsize=14)
plt.plot(fq, F3_abs_amp, c='orange')
# 振幅強度でフィルタリング(ノイズ除去) => IFFT
F3_ifft = np.fft.ifft(F3)
F3_ifft_real = F3_ifft.real # 実数部の取得
# グラフ
plt.plot(t,f,label='original')
plt.plot(t,F3_ifft_real, c='orange', linewidth=4, alpha=0.7, label='filtered')
plt.legend(loc="best")
plt.xlabel('time(sec)',fontsize=14)
plt.ylabel('signal',fontsize=14)
