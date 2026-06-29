"""Training-history, coverage, and xi-curve plots."""

import itertools
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from getdist import MCSamples

# colorblind-safe palette, no red/green (Wong 2011 minus its green
# and vermillion): blue, orange, reddish-purple, black, sky-blue.
_CB = ["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"]


def _finish(fig, savepath):
  """Save the figure and close, or show it.

  If savepath is given, write the figure there (format inferred
  from the extension, e.g. .pdf) and close it -- a batch script has
  no display to show on; if None, show it interactively.
  """
  if savepath is not None:
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
  else:
    plt.show()


def _history_panels(ax_loss, ax_frac, train_losses, medians,
                    means, fracs, thresholds):
  """
  Draw the two training-history panels onto the given axes.

  ax_loss gets train loss / val median / val mean vs epoch (log y,
  since the mean is heavy-tailed far above the median); ax_frac
  gets the fraction of val points over each delta-chi2 threshold vs
  epoch. Shared by plot_history (1x2) and plot_diagnostics (2x2),
  so the two never drift apart.

  Arguments:
    ax_loss, ax_frac = the two matplotlib axes to draw on.
    train_losses     = per-epoch training loss (list).
    medians / means  = per-epoch val median / mean chi2 (lists).
    fracs            = per-epoch list of 1D tensors; fracs[i] is
                       the fraction over each threshold at epoch i+1.
    thresholds       = 1D tensor of delta-chi2 cutoffs (labels).
  """
  epochs = range(1, len(medians) + 1)
  # Here x = epoch numbers and y = the per-epoch curve:
  # train loss, val median, val mean.
  ax_loss.semilogy(epochs,
                   train_losses,
                   color=_CB[0],
                   label="train")
  ax_loss.semilogy(epochs,
                   medians,
                   color=_CB[1],
                   label="val median")
  ax_loss.semilogy(epochs,
                   means,
                   color=_CB[2],
                   label="val mean")
  ax_loss.set_xlabel("epoch")
  ax_loss.set_ylabel("loss")
  ax_loss.legend(frameon=False)

  fr = torch.stack(fracs).cpu()      # (nepochs, n_thr)
  for j, t in enumerate(thresholds.tolist()):
    # x = epochs, y = the fraction of val points over threshold j.
    ax_frac.plot(epochs,
                 fr[:, j],
                 color=_CB[j % len(_CB)],
                 label=f"> {t:g}")
  ax_frac.set_xlabel("epoch")
  ax_frac.set_ylabel("fraction of val points")
  ax_frac.legend(frameon=False, title="delta chi2")


def _coverage_panels(ax_scatter, ax_hist, knn_dist, dchi2, k_nn):
  """
  Draw the two coverage-diagnostic panels onto the given axes.

  ax_scatter: per-val hardness log10(dchi2) vs local sparsity (mean
  distance to the k nearest training points), with the 0.2 goal
  line. ax_hist: the sparsity distributions of the good
  (dchi2<=0.2) and bad (dchi2>0.2) populations -- a right-shifted
  "bad" histogram means failures live where training is scarce.

  Arguments:
    ax_scatter, ax_hist = the two matplotlib axes to draw on.
    knn_dist = (Nval,) mean distance to the k nearest train points.
    dchi2    = (Nval,) per-val delta-chi2 (same row order).
    k_nn     = the k used (for the axis labels).
  """
  y   = np.log10(np.maximum(dchi2, 1e-4))
  bad = dchi2 > 0.2
  # (a) hardness vs local sparsity. x = knn_dist, y = log10 dchi2,
  # color = the same y; the dashed line is the 0.2 goal.
  sc = ax_scatter.scatter(knn_dist, y, s=5, c=y, cmap="viridis")
  ax_scatter.axhline(np.log10(0.2), color="0.4", lw=1, ls="--")
  ax_scatter.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_scatter.set_ylabel(r"$\log_{10}\,\Delta\chi^2$")
  # ax.figure is the parent figure -- add the colorbar to it.
  ax_scatter.figure.colorbar(sc,
                             ax=ax_scatter,
                             label=r"$\log_{10}\,\Delta\chi^2$")
  # (b) good vs bad sparsity. x = knn_dist, y = density; shared
  # bins so the two histograms are directly comparable.
  bins = np.linspace(knn_dist.min(), knn_dist.max(), 40)
  ax_hist.hist(knn_dist[~bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[0],
               label="good (dchi2<0.2)")
  ax_hist.hist(knn_dist[bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[1],
               label="bad (dchi2>0.2)")
  ax_hist.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_hist.set_ylabel("density")
  ax_hist.legend(frameon=False)


def plot_history(train_losses,
                 medians,
                 means,
                 fracs,
                 thresholds,
                 savepath=None):
  """
  Plot a run_emulator training history (the two history panels).

  Left: train loss, val median, val mean vs epoch (log y). Right:
  fraction of val points over each delta-chi2 threshold vs epoch.

  Arguments (the four histories run_emulator returns, plus the
  thresholds used):
    train_losses = per-epoch training loss (list of floats); the
                   sqrt-trimmed objective, so on a different scale
                   than the raw-chi2 val metrics.
    medians      = per-epoch val median chi2 (list).
    means        = per-epoch val mean chi2 (list).
    fracs        = per-epoch list of 1D tensors; fracs[i] holds the
                   fraction of val points over each threshold at
                   epoch i+1.
    thresholds   = 1D tensor of delta-chi2 cutoffs used in
                   training; labels the right panel.
    savepath     = if given, write the figure there (format from
                   the extension, e.g. .pdf) and close it; if None
                   (default), show it interactively.
  """
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  _history_panels(ax[0], ax[1], train_losses, medians, means,
                  fracs, thresholds)
  fig.tight_layout()
  _finish(fig, savepath)


def _floor_panel(ax, floor):
  """
  Draw the local-linear data-floor panel onto the given axis.

  Per val point: the model's delta-chi2 vs the data-only floor (the
  local-linear prediction's delta-chi2), log-log. Points on the
  diagonal mean the net is at what a smooth local method extracts
  from the data (data-limited); points well above mean the net has
  headroom. The dotted lines mark the 0.2 goal on each axis.

  Arguments:
    ax    = the matplotlib axis to draw on.
    floor = the dict local_linear_floor returned (dchi2_floor,
            dchi2_model, f_floor, f_model, f_hard).
  """
  lo = 1e-3
  dchi2_floor = floor["dchi2_floor"]
  dchi2_model = floor["dchi2_model"]
  # x = data-only floor dchi2, y = model dchi2; floor both at lo so
  # a near-zero point stays on the log axes.
  ax.scatter(np.maximum(dchi2_floor, lo),
             np.maximum(dchi2_model, lo),
             s=5, alpha=0.4, color=_CB[0])
  mx = max(dchi2_floor.max(), dchi2_model.max())
  ax.plot([lo, mx], [lo, mx], "k--", lw=1)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.axhline(0.2, color="0.6", lw=1, ls=":")
  ax.axvline(0.2, color="0.6", lw=1, ls=":")
  ax.set_xlabel(r"data-only $\Delta\chi^2$ (local-linear floor)")
  ax.set_ylabel(r"model $\Delta\chi^2$")
  ax.set_title(f"f_model {floor['f_model']:.3f}  vs  "
               f"f_floor {floor['f_floor']:.3f}  "
               f"(pure hardness {floor['f_hard']:.3f})")


def _hard_direction_panels(ax_uni, ax_joint, hd):
  """
  Draw the hard-direction regression onto two axes (bar charts).

  ax_uni: each feature's univariate |correlation| with log10 dchi2
  (a collinearity-robust ranking). ax_joint: the joint log-linear
  OLS coefficients (the alpha, beta, ... combination). The joint
  R^2 and the ln(omega_b h^2)-alone R^2 are in the titles. Both
  panels share the feature order (descending univariate |corr|).

  Arguments:
    ax_uni, ax_joint = the two matplotlib axes to draw on.
    hd = the dict hard_direction_regression returned (labels,
         univariate, joint_coef, r2, r2_omega).
  """
  labels = hd["labels"]
  uni    = hd["univariate"]
  coef   = hd["joint_coef"]
  # order features by descending univariate |corr|; both panels use
  # this same order so the bars line up.
  order = np.argsort(np.abs(uni))[::-1]
  ypos  = np.arange(len(order))
  names = [labels[j] for j in order]

  # barh(y, width): y = bar slot, width = the value.
  ax_uni.barh(ypos, np.abs(uni)[order], color=_CB[0])
  ax_uni.set_yticks(ypos)
  ax_uni.set_yticklabels(names)
  ax_uni.invert_yaxis()                 # strongest feature at top
  ax_uni.set_xlabel("univariate |corr| with log10 dchi2")
  ax_uni.set_title("univariate ranking")

  ax_joint.barh(ypos, coef[order], color=_CB[1])
  ax_joint.set_yticks(ypos)
  ax_joint.set_yticklabels(names)
  ax_joint.invert_yaxis()
  ax_joint.axvline(0.0, color="0.6", lw=1)
  ax_joint.set_xlabel("joint log-linear coefficient")
  ax_joint.set_title(f"joint R2 {hd['r2']:.3f}  |  "
                     f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")


def _save_pages(figs, savepath):
  """
  Save figures as a multipage PDF, or show them.

  If savepath is given, write every figure as one page of a single
  PDF (matplotlib's PdfPages) and close them -- a batch script has
  no display; if None, show them interactively.

  Arguments:
    figs     = list of matplotlib Figures, one per page.
    savepath = the .pdf path, or None to show.
  """
  if savepath is None:
    plt.show()
    return
  from matplotlib.backends.backend_pdf import PdfPages
  with PdfPages(savepath) as pdf:
    for f in figs:
      pdf.savefig(f, bbox_inches="tight")
      plt.close(f)


def plot_diagnostics(train_losses,
                     medians,
                     means,
                     fracs,
                     thresholds,
                     coverage,
                     floor=None,
                     hard_dir=None,
                     savepath=None):
  """
  All available diagnostics as a single MULTIPAGE figure / PDF.

  Page 1 (2x2): the training history (loss curves; fraction over
    each delta-chi2 threshold vs epoch) AND the coverage diagnostic
    (hardness vs local sparsity; good/bad sparsity histograms).
  Page 2: the local-linear data-only floor (model vs floor
    delta-chi2), if `floor` is given.
  Page 3: the hard-direction regression (univariate ranking and the
    joint log-linear coefficients), if `hard_dir` is given.

  floor / hard_dir are optional so a run can drop a page it cannot
  produce (e.g. the local-linear floor is only defined for a plain
  chi2fn, so a --rescale run omits it).

  Arguments:
    train_losses, medians, means, fracs, thresholds = the
      run_emulator histories (see plot_history).
    coverage = the dict coverage_diagnostic returned.
    floor    = the dict local_linear_floor returned, or None.
    hard_dir = the dict hard_direction_regression returned, or None.
    savepath = if given, write a (multipage) PDF there and close;
               if None, show each page interactively.
  """
  figs = []
  # page 1: history (top row) + coverage (bottom row).
  f1, ax = plt.subplots(2, 2, figsize=(12, 9))
  _history_panels(ax[0, 0], ax[0, 1], train_losses, medians,
                  means, fracs, thresholds)
  _coverage_panels(ax[1, 0], ax[1, 1], coverage["knn_dist"],
                   coverage["dchi2"], coverage["k_nn"])
  f1.tight_layout()
  figs.append(f1)
  # page 2: the local-linear data floor (plain chi2fn only).
  if floor is not None:
    f2, a2 = plt.subplots(figsize=(6, 6))
    _floor_panel(a2, floor)
    f2.tight_layout()
    figs.append(f2)
  # page 3: the hard-direction regression.
  if hard_dir is not None:
    f3, a3 = plt.subplots(1, 2, figsize=(13, 6))
    _hard_direction_panels(a3[0], a3[1], hard_dir)
    f3.tight_layout()
    figs.append(f3)
  _save_pages(figs, savepath)


def source_param_samples(source, names, labels, label):
  """
  getdist MCSamples of one source's cosmological parameters.

  Pulls the rows the source actually uses (source["idx"]) from
  its parameter dump and wraps them as equally-weighted
  samples for a coverage triangle (no likelihood, no chi2).
  Reads no module globals -- source, names, labels, and the
  legend label all arrive as arguments.

  Arguments:
    source = source dict with "C" (full param dump) and "idx"
             (global rows actually in use).
    names  = parameter column names, in the dump's column
             order (pgeom.names).
    labels = LaTeX labels for those columns (no surrounding $).
    label  = legend label for this set (e.g. "train").

  Returns:
    an MCSamples over the source's used parameter rows.
  """
  # the rows this source actually uses -- coverage is about
  # what was trained / validated on, not the whole file.
  rows = np.sort(source["idx"])
  # raw physical parameters of those rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  return MCSamples(samples=P, 
                   names=names, 
                   labels=labels,
                   label=label,
                   settings={"smooth_scale_1D": 0.3,
                             "smooth_scale_2D": 0.3,
                             "fine_bins_2D": 512})


def dv_to_xi(dv_row, geom):
  """
  Reshape one full data-vector row into the (theta, xip, xim)
  matrix layout of plot_xi, using its cosmic-shear block.

  Takes the leading xi_size entries (xi_plus then xi_minus,
  pairs (i<=j) outer / theta inner) and scatters each pair's
  ntheta values into the (i, j) slot of an
  (ntheta, ntomo, ntomo) array (upper triangle filled; the
  rest stay 0 and plot_xi never reads them).

  Arguments:
    dv_row = (total_size,) full data vector; only the leading
             geom.xi_size cosmic-shear entries are used.
    geom   = geometry carrying ntheta / source_ntomo /
             theta_centers / xi_size.
  Returns:
    (theta, xip, xim): theta (ntheta,) [arcmin], xip/xim (ntheta, ntomo, ntomo).
  """
  nt    = geom.source_ntomo
  ntha  = geom.ntheta
  block = np.asarray(dv_row[:geom.xi_size], dtype="float64")
  pairs = [(i, j) for i in range(nt) for j in range(i, nt)]
  half  = len(pairs) * ntha
  xip = np.zeros((ntha, nt, nt))
  xim = np.zeros((ntha, nt, nt))
  for p, (i, j) in enumerate(pairs):
    xip[:, i, j] = block[p * ntha:(p + 1) * ntha]
    xim[:, i, j] = block[half + p * ntha:
                         half + (p + 1) * ntha]
  return (geom.theta_centers, xip, xim)


def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None, 
            marker = None, linestyle = None, linewidth = None, 
            ylim = [0.88,1.12], cmap = 'gist_rainbow', legend = None, 
            legendloc = (0.6,0.78), yaxislabelsize = 16, yaxisticklabelsize = 10, 
            xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]], 
            bintextsize = 15, figsize = (12, 12), show = None, thetashow=[3,1000], 
            colorbar=1):
    
    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    if xi_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0})    

    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        norm = matplotlib.colors.Normalize(vmin=param[0],
                                           vmax=param[-1])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.5
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, 
                         size = 20, 
                         weight = 'bold', 
                         labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), 
                                        np.max(ylim[1]*np.array(ximax))])
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')
                
                if i == 0:
                    if xi_ref is None:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", 
                                                 fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if xi_ref is None:
                    # plot(x, y, ...): x = theta, y = theta *
                    # xi_+/- * 1e4 (the scaled correlation fn).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    # plot(x, y, ...): x = theta, y = xi_+/- /
                    # xi_ref - 1 (the fractional difference).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if not np.array_equal(theta, theta_ref):
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None',
                                               marker=next(markercycler),  
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, 
                                                       xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                                       color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), 
                                                       linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)    
    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        fig.legend(legend, 
                   loc=legendloc,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    if not (show is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.show()
    else:
        return (fig, axes)
