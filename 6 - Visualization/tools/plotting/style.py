import matplotlib
# Force headless Agg backend for SSH/headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_style():
    # prefer seaborn-darkgrid if available, otherwise fall back to default style
    try:
        plt.style.use('seaborn-darkgrid')
    except Exception:
        try:
            plt.style.use('default')
        except Exception:
            pass
    return plt
