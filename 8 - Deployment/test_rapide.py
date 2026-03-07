import spidev, time
import numpy as np
from scipy.fft import rfft, rfftfreq
import time

def precise_sleep(dt):
    end = time.perf_counter() + dt
    while time.perf_counter() < end:
        pass

spi = spidev.SpiDev()
spi.open(0, 0)                 # bus 0, CE0
spi.max_speed_hz = int(1000e3)  # 1 MHz (safe)
spi.mode = 1                   # ou 3
spi.no_cs = False              # IMPORTANT: laisser le kernel gérer CS

def read_mcp3001():
    rx = spi.xfer2([0x00, 0x00])              # 16 clocks
    val = ((rx[0] & 0x1F) << 5) | (rx[1] >> 3)
    return rx, val

N = 200000
f = 24000 # Hz
times = np.zeros(N)
timesAlt = np.zeros(N)
data = np.zeros(N)
dt = 1/f

for i in range(N):
    rx, v = read_mcp3001()
    data[i] = v
    
    end = timesAlt[i-1] + dt if i > 0 else time.perf_counter() + dt
    while time.perf_counter() < end:
        pass
    timesAlt[i] = time.perf_counter() #end
    times[i] = time.perf_counter()
    
    print(f"rx={rx[0]:02X} {rx[1]:02X}  val={v} " + "|"*(v//5))
    #time.sleep(1/f)
    #precise_sleep(1/f)

lf = rfftfreq(N, d=1/f)
data = data - np.mean(data)
fft = rfft(data)
FILTER = (lf > 200)
main_freq = lf[(FILTER * abs(fft)).argmax()]
print(main_freq)

def plot_fft():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(lf, np.abs(fft))
    plt.title('FFT of Captured Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, f/2)
    plt.grid()
    plt.savefig('test_recorder_pi_fft.png')
    print('Saved FFT plot to test_recorder_pi_fft.png')
    
plot_fft()

print(f"Fréquence réelle: {N/(times[-1] - times[0])} Hz sur durée {times[-1] - times[0]:.3f} s")
print(f"Fréquence réelle (temps alternatif): {N/(timesAlt[-1] - timesAlt[0])} Hz sur durée {timesAlt[-1] - timesAlt[0]:.3f} s")
