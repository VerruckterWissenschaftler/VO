import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] quaternion → [roll, pitch, yaw] (ZYX, radians)."""
    x, y, z, w = q
    roll  = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    sinp  = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([roll, pitch, yaw])


def _interp_cols(target_t: np.ndarray, src_t: np.ndarray,
                 src_vals: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(target_t, src_t, src_vals[:, i])
        for i in range(src_vals.shape[1])
    ])


# ──────────────────────────────────────────────────────────────────────────────
# UKFDebugger
# ──────────────────────────────────────────────────────────────────────────────
# State layout (12-state UKF):
#   x = [px, py, pz,  vx, vy, vz,  roll, pitch, yaw,  wx, wy, wz]
#         0   1   2    3   4   5    6     7      8     9  10  11

_POS   = slice(0, 3)
_VEL   = slice(3, 6)
_ANG   = slice(6, 9)
_OMEGA = slice(9, 12)


class UKFDebugger:
    """
    Attaches to a UKF instance and records per-step internal state for
    offline inspection.

    Usage
    -----
        dbg = UKFDebugger(ukf)
        vo_estimator.run()
        dbg.summary()
        dbg.plot(gt_timestamps, gt_positions, gt_quaternions)
    """

    def __init__(self, ukf):
        self.ukf = ukf
        self._predict_log: list[dict] = []
        self._update_log:  list[dict] = []
        self._cumtime: float = 0.0
        self._attach()

    # ── Patching ──────────────────────────────────────────────────────────────

    def _attach(self):
        orig_predict    = self.ukf.predict
        orig_vo_update  = self.ukf.vo_update
        orig_gyro       = self.ukf.gyro_update
        orig_append_ori = self.ukf.append_orientation
        dbg = self

        def patched_predict(dt):
            orig_predict(dt)
            if dbg.ukf.initialized and dt > 0:
                dbg._cumtime += dt
            dbg._record_predict()

        def patched_vo_update(position):
            inn = np.zeros(3)
            if dbg.ukf.initialized:
                inn = position - dbg.ukf.x[_POS]
            orig_vo_update(position)
            dbg._record_update(inn, 'vo')

        def patched_gyro(omega_meas):
            inn = np.zeros(3)
            if dbg.ukf.initialized:
                inn = omega_meas - dbg.ukf.x[_OMEGA]
            orig_gyro(omega_meas)
            dbg._record_update(inn, 'gyro')

        def patched_append_ori(R):
            orig_append_ori(R)
            dbg._record_update(np.zeros(3), 'ori')

        self.ukf.predict           = patched_predict
        self.ukf.vo_update         = patched_vo_update
        self.ukf.gyro_update       = patched_gyro
        self.ukf.append_orientation = patched_append_ori

    def detach(self):
        for name in ('predict', 'vo_update', 'gyro_update', 'append_orientation'):
            try:
                delattr(self.ukf, name)
            except AttributeError:
                pass

    # ── Recording ─────────────────────────────────────────────────────────────

    def _record_predict(self):
        if not self.ukf.initialized:
            return
        self._predict_log.append({
            'step':    len(self._predict_log),
            'cumtime': self._cumtime,
            'x':       self.ukf.x.copy(),
            'a_world': self.ukf._world_accel(),
        })

    def _record_update(self, innovation: np.ndarray, kind: str):
        if not self.ukf.initialized:
            return
        self._update_log.append({
            'step':       len(self._predict_log) - 1,
            'innovation': innovation.copy(),
            'inn_norm':   float(np.linalg.norm(innovation)),
            'kind':       kind,
        })

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        if not self._predict_log:
            print("UKFDebugger: no data recorded.")
            return

        x_arr  = np.array([r['x'] for r in self._predict_log])
        n_vo   = sum(1 for r in self._update_log if r['kind'] == 'vo')
        n_gyro = sum(1 for r in self._update_log if r['kind'] == 'gyro')
        n_ori  = sum(1 for r in self._update_log if r['kind'] == 'ori')

        groups = [
            ('Position (m)',        _POS,   ['px',    'py',    'pz'   ]),
            ('Lin. velocity (m/s)', _VEL,   ['vx',    'vy',    'vz'   ]),
            ('Euler angles (rad)',  _ANG,   ['roll',  'pitch', 'yaw'  ]),
        ]

        print("=" * 54)
        print("  UKF Debug Summary")
        print("=" * 54)
        print(f"  Predict steps  : {len(self._predict_log)}")
        print(f"  Duration       : {self._cumtime:.2f} s")
        print(f"  VO updates     : {n_vo}")
        print(f"  Gyro updates   : {n_gyro}")
        print(f"  Ori updates    : {n_ori}")
        for title, sl, names in groups:
            vals = x_arr[:, sl]
            print(f"  {title}")
            for i, name in enumerate(names):
                v = vals[:, i]
                print(f"    {name:>6s}:  mean={v.mean(): .4f}  "
                      f"std={v.std():.4f}  [{v.min():.4f}, {v.max():.4f}]")
        vo_log = [r for r in self._update_log if r['kind'] == 'vo']
        if vo_log:
            norms = [r['inn_norm'] for r in vo_log]
            print(f"\n  VO innovation norms:  mean={np.mean(norms):.4f}"
                  f"  max={np.max(norms):.4f}")
        print("=" * 54)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot(self,
             gt_timestamps:  np.ndarray | None = None,
             gt_positions:   np.ndarray | None = None,
             gt_quaternions: np.ndarray | None = None,
             max_steps:      int        | None = None):
        """
        5 × 4 per-direction diagnostic figure with slider + Prev/Next.

        Layout
        ------
        Row 0  pos x | pos y | pos z       | position error (m)
        Row 1  vel x | vel y | vel z       | innovation norms (VO)
        Row 2  roll  | pitch | yaw         | euler angle error (deg)
        Row 3  GT acc x | GT acc y | GT acc z | [live info text]
        Row 4  ax_w  | ay_w  | az_w        | (world-frame accel, medfilt)
        """
        if not self._predict_log:
            print("UKFDebugger: no data to plot.")
            return

        log   = self._predict_log[:max_steps] if max_steps else self._predict_log
        n     = len(log)
        steps = np.arange(n)

        x_arr  = np.array([r['x']       for r in log])   # (n, 12)
        ct_arr = np.array([r['cumtime'] for r in log])   # (n,)
        a_arr  = np.array([r['a_world'] for r in log])   # (n, 3)

        # ── GT interpolation ──────────────────────────────────────────────────
        gt_pos_interp   = None
        gt_euler_interp = None
        pos_err         = None
        euler_err_deg   = None

        if gt_timestamps is not None and gt_positions is not None and len(gt_timestamps) > 1:
            gt_t = np.asarray(gt_timestamps, dtype=float)
            gt_t = gt_t - gt_t[0]
            gt_pos_interp = _interp_cols(ct_arr, gt_t, np.asarray(gt_positions))
            pos_err = np.linalg.norm(x_arr[:, _POS] - gt_pos_interp, axis=1)

            if gt_quaternions is not None and len(gt_quaternions) == len(gt_timestamps):
                gt_e = np.array([_quat_to_euler(q) for q in gt_quaternions])
                gt_euler_interp = _interp_cols(ct_arr, gt_t, gt_e)
                raw_diff = x_arr[:, _ANG] - gt_euler_interp
                euler_err_deg = np.degrees(
                    np.abs((raw_diff + np.pi) % (2 * np.pi) - np.pi)
                )

        has_gt       = gt_pos_interp is not None
        has_gt_euler = gt_euler_interp is not None

        # ── GT velocity (finite difference, interpolated to UKF steps) ────────
        gt_vel_interp = None
        if has_gt and len(gt_t) > 1:
            gt_dt  = np.diff(gt_t)
            gt_dt  = np.maximum(gt_dt, 1e-9)
            gt_vel = np.diff(np.asarray(gt_positions), axis=0) / gt_dt[:, None]
            gt_vel_t = gt_t[1:]
            gt_vel_interp = _interp_cols(ct_arr, gt_vel_t, gt_vel)

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle("UKF Debugger", fontsize=13, fontweight='bold')

        gs = GridSpec(5, 4, figure=fig,
                      top=0.94, bottom=0.16,
                      hspace=0.55, wspace=0.38)

        vlines: list = []

        def _ax(row, col, title, ylabel=''):
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.set_xlabel('step', fontsize=7)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, linewidth=0.4)
            vl = ax.axvline(0, color='red', linewidth=1.0, alpha=0.8, zorder=10)
            vlines.append(vl)
            return ax

        C  = 'tab:blue'
        GT = 'tab:green'

        # ── Row 0: Position ───────────────────────────────────────────────────
        for col, (lbl, idx) in enumerate([('pos x', 0), ('pos y', 1), ('pos z', 2)]):
            ax = _ax(0, col, lbl, 'm')
            ax.plot(steps, x_arr[:, idx], color=C, linewidth=0.9, label='UKF')
            if has_gt:
                ax.plot(steps, gt_pos_interp[:, col],
                        color=GT, linewidth=0.9, linestyle='--', label='GT')
                ax.legend(fontsize=6, loc='upper right')

        ax_pos_err = _ax(0, 3, 'pos error (m)', 'm')
        if has_gt:
            ax_pos_err.plot(steps, pos_err, color='tab:red', linewidth=0.9, label='|err|')
            for i, lbl in enumerate(['x', 'y', 'z']):
                ax_pos_err.plot(steps,
                                np.abs(x_arr[:, i] - gt_pos_interp[:, i]),
                                linewidth=0.6, alpha=0.6, label=lbl)
            ax_pos_err.legend(fontsize=6, loc='upper right')
        else:
            ax_pos_err.text(0.5, 0.5, 'No GT', ha='center', va='center',
                            transform=ax_pos_err.transAxes, color='grey', fontsize=9)

        # ── Row 1: Linear velocity ────────────────────────────────────────────
        for col, (lbl, idx) in enumerate([('vel x', 3), ('vel y', 4), ('vel z', 5)]):
            ax = _ax(1, col, lbl, 'm/s')
            ax.plot(steps, x_arr[:, idx], color=C, linewidth=0.9, label='UKF')
            if gt_vel_interp is not None:
                ax.plot(steps, gt_vel_interp[:, col],
                        color=GT, linewidth=0.9, linestyle='--', label='GT')
                ax.legend(fontsize=6, loc='upper right')

        ax_inn = _ax(1, 3, 'VO innovation norms', 'm')
        vo_log = [r for r in self._update_log if r['kind'] == 'vo']
        if vo_log:
            vo_s = [r['step']     for r in vo_log]
            vo_n = [r['inn_norm'] for r in vo_log]
            ax_inn.scatter(vo_s, vo_n, s=8, color='tab:blue', label='VO', zorder=3, alpha=0.8)
            ax_inn.legend(fontsize=6, loc='upper right')
        else:
            ax_inn.text(0.5, 0.5, 'No VO updates', ha='center', va='center',
                        transform=ax_inn.transAxes, color='grey', fontsize=9)

        # ── Row 2: Euler angles ───────────────────────────────────────────────
        for col, (lbl, idx) in enumerate([('roll', 6), ('pitch', 7), ('yaw', 8)]):
            ax = _ax(2, col, lbl, 'rad')
            ax.plot(steps, x_arr[:, idx], color=C, linewidth=0.9, label='UKF')
            if has_gt_euler:
                ax.plot(steps, gt_euler_interp[:, col],
                        color=GT, linewidth=0.9, linestyle='--', label='GT')
                ax.legend(fontsize=6, loc='upper right')

        ax_euler_err = _ax(2, 3, 'euler error', 'deg')
        if has_gt_euler:
            for i, lbl in enumerate(['roll', 'pitch', 'yaw']):
                ax_euler_err.plot(steps, euler_err_deg[:, i], linewidth=0.8, label=lbl)
            ax_euler_err.legend(fontsize=6, loc='upper right')
        else:
            ax_euler_err.text(0.5, 0.5, 'No GT', ha='center', va='center',
                              transform=ax_euler_err.transAxes, color='grey', fontsize=9)

        # ── Row 3: GT acceleration (finite-diff of GT velocity) ──────────────
        gt_acc_interp = None
        if gt_vel_interp is not None and len(gt_vel_t) > 1:
            gt_acc_dt = np.diff(gt_vel_t)
            gt_acc_dt = np.maximum(gt_acc_dt, 1e-9)
            gt_acc    = np.diff(gt_vel, axis=0) / gt_acc_dt[:, None]
            gt_acc_t  = gt_vel_t[1:]
            gt_acc_interp = _interp_cols(ct_arr, gt_acc_t, gt_acc)

        for col, lbl in enumerate(['GT acc x', 'GT acc y', 'GT acc z']):
            ax = _ax(3, col, lbl, 'm/s²')
            if gt_acc_interp is not None:
                ax.plot(steps, gt_acc_interp[:, col], color=GT, linewidth=0.9)
            else:
                ax.text(0.5, 0.5, 'No GT', ha='center', va='center',
                        transform=ax.transAxes, color='grey', fontsize=9)

        # ── Row 4: World-frame acceleration (median-filtered) ─────────────────
        AC = 'tab:orange'
        for col, lbl in enumerate(['ax world', 'ay world', 'az world']):
            ax = _ax(4, col, lbl, 'm/s²')
            ax.plot(steps, a_arr[:, col], color=AC, linewidth=0.9)
        ax_a_info = fig.add_subplot(gs[4, 3])
        ax_a_info.axis('off')
        ax_a_info.text(0.5, 0.5, 'world-frame accel\n(medfilt, gravity removed)',
                       ha='center', va='center', transform=ax_a_info.transAxes,
                       fontsize=8, color='grey')

        # ── Row 3 col 3: live info text ───────────────────────────────────────
        info_ax = fig.add_subplot(gs[3, 3])
        info_ax.axis('off')
        info_text = info_ax.text(
            0.04, 0.97, '', transform=info_ax.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        )

        def _build_info(s: int) -> str:
            s  = int(np.clip(s, 0, n - 1))
            x  = log[s]['x']
            t  = log[s]['cumtime']
            a  = log[s]['a_world']
            lines = [
                f"step {s}  t={t:.3f}s",
                f"pos  [{x[0]:+.3f} {x[1]:+.3f} {x[2]:+.3f}]",
                f"vel  [{x[3]:+.3f} {x[4]:+.3f} {x[5]:+.3f}]",
                f"rpy  [{x[6]:+.3f} {x[7]:+.3f} {x[8]:+.3f}]",
                f"acc  [{a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}]",
            ]
            if has_gt:
                gp = gt_pos_interp[s]
                lines.append(f"GT p [{gp[0]:+.3f} {gp[1]:+.3f} {gp[2]:+.3f}]")
                lines.append(f"perr  {pos_err[s]:.4f} m")
            if gt_vel_interp is not None:
                gv = gt_vel_interp[s]
                lines.append(f"GT v [{gv[0]:+.3f} {gv[1]:+.3f} {gv[2]:+.3f}]")
            if gt_acc_interp is not None:
                ga = gt_acc_interp[s]
                lines.append(f"GT a [{ga[0]:+.3f} {ga[1]:+.3f} {ga[2]:+.3f}]")
            if has_gt_euler:
                ge = gt_euler_interp[s]
                ed = euler_err_deg[s]
                lines.append(f"GT e [{ge[0]:+.3f} {ge[1]:+.3f} {ge[2]:+.3f}]")
                lines.append(f"eerr [{ed[0]:.2f} {ed[1]:.2f} {ed[2]:.2f}]°")
            return '\n'.join(lines)

        info_text.set_text(_build_info(0))

        # ── Slider + Prev / Next ──────────────────────────────────────────────
        ax_prev   = fig.add_axes([0.10, 0.06, 0.05, 0.025])
        ax_slider = fig.add_axes([0.17, 0.06, 0.62, 0.025])
        ax_next   = fig.add_axes([0.81, 0.06, 0.05, 0.025])

        btn_prev = Button(ax_prev,   '< Prev')
        slider   = Slider(ax_slider, 'Step', 0, n - 1, valinit=0, valstep=1)
        btn_next = Button(ax_next,   'Next >')

        def on_slider(val):
            s = int(slider.val)
            for vl in vlines:
                vl.set_xdata([s, s])
            info_text.set_text(_build_info(s))
            fig.canvas.draw_idle()

        def on_prev(_):
            slider.set_val(int(np.clip(slider.val - 1, 0, n - 1)))

        def on_next(_):
            slider.set_val(int(np.clip(slider.val + 1, 0, n - 1)))

        slider.on_changed(on_slider)
        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        plt.show()
        return fig
