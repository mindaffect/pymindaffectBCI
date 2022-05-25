#  Copyright (c) 2019 MindAffect B.V.
#  Author: Jason Farquhar <jadref@gmail.com>
# This file is part of pymindaffectBCI <https://github.com/mindaffect/pymindaffectBCI>.
#
# pymindaffectBCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pymindaffectBCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pymindaffectBCI.  If not, see <http://www.gnu.org/licenses/>

import numpy as np
from mindaffectBCI.decoder.utils import window_axis


#@function
def scoreStimulus(X, W, R=None, b=None, offset=0, f=None, isepoched=False):
    '''
    Apply spatio-temporal (possibly factored) model to data 

    Args:
      X_TSd (ndarray): (nTrl,nSamp,d) raw response for the current stimulus event
         OR    X_TEtd (nTrl,nEpoch,tau,d) pre-sliced raw data
      W_Mkd (ndarray): (nModel,rank,d) spatial filters for each output
      R_Mket (ndarray): (nModel,rank,nEvent,tau) responses for each stimulus event for each output
        OR
      W_Metd (ndarray): (nModel,nEvent,tau,d) spatio-temporal filter per event type and model
      R (None)
      b_e : (nEvent) bias for each stimulus type
      f_f : (nFreq) spectral filter weighting
      offset (int): offset in X for applying W

    Returns
      Fe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    Copyright (c) MindAffect B.V. 2018
    '''
    tau = W.shape[-2] if R is None else R.shape[-1]  # est response length
    if isepoched is None:  # guess epoched state from the shapes...
        isepoched = X.ndim > 3
    if isepoched:
        Fe = scoreStimulusEpoch(X, W, R, b)
    else:
        Fe = scoreStimulusCont(X, W, R, b, offset)
    return Fe


def scoreStimulusEpoch(X, W, R=None, b=None):
    '''
    Apply spatio-temporal (possibly factored) model to epoched data 
      X = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nfilt x d) spatial filters for each output
      R = (nM x nfilt x nE x tau) responses for each stimulus event for each output
     OR
      W = (nM x nE x tau x d) spatio-temporal filter per event type and model
      R = None
      b = (nE,1) offset for each stimulus type
    Outputs:
      Fe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    '''
    if R is not None:
        Fe = scoreStimulusEpoch_factored(X, W, R, b)
    else:
        Fe = scoreStimulusEpoch_full(X, W, b)
    return Fe


def scoreStimulusEpoch_factored(X, W, R, b=None):
    '''
    Apply factored spatio-temporal model to epoched data 
      X = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W = (nM x nfilt x d) spatial filters for each output
      R = (nM x nfilt x nE x tau) responses for each stimulus event for each output
      b = (nE,1) offset for each stimulus type
    Outputs:
      Fe= (nM x nTrl x nEpoch x nE) similarity score for each input epoch for each output
    '''

    # ensure all inputs have the right  shape, by addig leading singlenton dims
    X = X.reshape((1, ) * (4 - X.ndim) + X.shape)  # (nTrl,nEp,tau,d)
    W = W.reshape(((1, ) * (3 - W.ndim)) + W.shape)  # (nM,nfilt,d)
    R = R.reshape(((1, ) * (4 - R.ndim)) + R.shape)  # (nM,nfile,nE,tau)

    # apply the factored model
    # N.B. einsum seems to mess the ram up, so do it ourselves...
    #Fe = np.einsum("Mkd,TEtd,Mket->MTEe", W, X, R, optimize='optimal')
    Fe = np.einsum("Mkd,TEtd->TEMkt", W, X)  # manual factored
    Fe = np.einsum("TEMkt,Mket->MTEe", Fe, R)
    if not b is None:  # include the bias, for each stimulus type
        Fe = Fe + b
    return Fe


def scoreStimulusEpoch_full(X_TStd, W_Metd, b=None):
    '''
    Apply full spatio-temporal model to epoched data 
      X_TStd = (nTrl x nEpoch x tau x d) pre-sliced raw data
      W_Metd = (nM x nE x tau x d) spatio-temporal filter per event type and model
      b_e = (nE,1) offset for each stimulus type
    Outputs:
      Fe_MTSe= (nM x nTrl x nEpoch/nSamp x nE) similarity score for each input epoch for each output
    '''

    # ensure inputs have the  right  shape
    X_TStd = X_TStd.reshape((1, ) * (4 - X_TStd.ndim) + X_TStd.shape)
    W_Metd = W_Metd.reshape((1, ) * (4 - W_Metd.ndim) + W_Metd.shape)

    # apply the model
    Fe_MTSe = np.einsum("TStd, Metd->MTSe", X_TStd, W_Metd, optimize='optimal')
    return Fe_MTSe


def factored2full(W, R):
    ''' convert a factored spatio-temporal model to a full model
    Inputs:
       W (nM,rank,d) spatial filter set (BWD model)
       R (nM,rank,e,tau) temporal filter set (FWD model)
    Output:
       W (nM,e,tau,d) spatio-temporal filter (BWD model) '''
    if R is not None:
        W = W.reshape(((1, ) * (3 - W.ndim)) + W.shape)
        R = R.reshape(((1, ) * (4 - R.ndim)) + R.shape)
        # get to single spatio-temporal filter
        W = np.einsum("mfd, mfet->metd", W, R)
    return W


def scoreStimulusCont(X_TSd, W, R=None, b=None, offset=0):
    """ Apply spatio-tempoal (possibly factored) model to raw (non epoched) data

    Args:
        X (np.ndarray (nTr,nSamp,d)): raw per-trial data
        W (np.ndarray (nM,nfilt,d)) OR (nM,nE,tau,d): spatial filters for each factor
        R (np.ndarray (nM,nfilt,nE,tau): responses for each stimulus event for each output
        b (np.ndarray (nE,1)): offset for each stimulus type
    Returns:
        np.ndarray (nM,nTrl,nSamp,nE): similarity score for each input epoch for each output
    """
    tau = W.shape[-2] if R is None else R.shape[-1]  # get response length
    if R is None:  # W = (nM, nE, tau, d)
        nM = W.shape[0] if W.ndim > 3 else 1
        nE = W.shape[-3] if W.ndim > 2 else 1
    else:
        nM = W.shape[0] if W.ndim > 2 else 1
        nE = R.shape[-2] if R.ndim > 1 else 1
    if X_TSd.shape[-2] < tau:  # X isn't big enough to apply... => zero score
        F_MTSe = np.zeros((nM, X_TSd.shape[0], X_TSd.shape[1], nE),
                          dtype=X_TSd.dtype)
        return F_MTSe

    # slice and apply
    # loopy apply as memory management is a thing!
    Fe_MTSe = np.zeros((nM, X_TSd.shape[0], X_TSd.shape[1], nE),
                       dtype=X_TSd.dtype)
    for ti in range(X_TSd.shape[0]):
        X_TStd = window_axis(X_TSd[ti:ti + 1, ...], winsz=tau,
                             axis=-2)  # (nTrl, nSamp-tau, tau, d)
        Fe_M1Se = scoreStimulusEpoch(X_TStd, W, R,
                                     b)  # (nM, nTrl, nSamp-tau, nE)

        # shift for the offset and zero-pad to the input X size
        # N.B. as we are offsetting from X->Y we move in the **OPPOSITTE** direction to
        # how Y is shifted!
        Feoffset = -offset
        if Feoffset <= 0:
            tmp = Fe_M1Se[..., -Feoffset:, :]  # shift-back and shrink
            Fe_M1Se = np.zeros(Fe_M1Se.shape[:-2] + (X_TSd.shape[-2], ) +
                               Fe_M1Se.shape[-1:],
                               dtype=Fe_MTSe.dtype)
            Fe_M1Se[..., :tmp.shape[-2], :] = tmp  # insert
        else:
            tmp = Fe_M1Se[..., :X_TSd.shape[-2] - Feoffset, :]  # shrink
            Fe_M1Se = np.zeros(Fe_M1Se.shape[:-2] + (X_TSd.shape[-2], ) +
                               Fe_M1Se.shape[-1:],
                               dtype=Fe_MTSe.dtype)
            Fe_M1Se[..., Feoffset:Feoffset +
                    tmp.shape[-2], :] = tmp  # shift + insert

        Fe_MTSe[:, ti, ...] = Fe_M1Se

    return Fe_MTSe


def scoreStimulus_temporal(X_TSd, W_kd, R_ket, b=None):
    '''
    Apply factored spatio-temporal model to epoched data 
      X_TSd : (nTrl,nSamp,d) pre-sliced raw data
      W None -- we learn this, remember?
      R_ket : (nfilt,nE,tau) responses for each stimulus event for each output
      b_Me : (nE,1) offset for each stimulus type
    Outputs:
      F_TSked : (nTrl, nSamp, nfilt, nE, d): similarity score for each input epoch for each output
    '''
    if W_kd is not None:
        raise ValueError("W should be None!")

    # ensure all inputs have the  right  shape
    X_TSd = X_TSd.reshape((1, ) * (3 - X_TSd.ndim) + X_TSd.shape)
    if R_ket.ndim == 4:
        print("Striping model dim!")
        R_ket = R_ket[0, ...]
    R_ket = R_ket.reshape((1, ) * (3 - R_ket.ndim) + R_ket.shape)

    # TODO[]: This is very memory inefficient -- rewrite...

    # window X into epochs
    X_TStd = window_axis(X_TSd, winsz=R_ket.shape[-1], axis=-2)

    # apply the temporal part of the model
    #print(X_TStd.shape)
    #print(R_Mket.shape)
    F_TSked = np.einsum("TStd,ket->TSked",
                        X_TStd,
                        R_ket,
                        dtype=X_TSd.dtype,
                        casting='unsafe')
    if not b is None:  # include the bias, for each stimulus type
        F_TSked = F_TSked + b[:, np.newaxis]

    # zero pad to make up to the right size
    padsz = list(F_TSked.shape)
    padsz[1] = X_TSd.shape[1] - X_TStd.shape[1]
    pad = np.zeros(padsz, dtype=X_TSd.dtype)
    F_TSked = np.append(F_TSked, pad, axis=1)

    return F_TSked


def dp(M_Tqp, beam_width=None, verb=0, weight=None):
    """dynamic programming solution extraction with beam_search for summation problems, that is:
       J = max_{w_p, I_Tq} M_Tqp * I_Tq * w_p = \sum_t M_Tqp[t,I_Tq[t]>0,:] @ w_p
    where
       M_Tqp is the reward for begin in state q at time T distributed over the p features
       I_Tq is a 0-1 indicator saying that state q is active at time T
       w_p is a weighting over the features p

    N.B. Assuming a *single* set of parameters for the *entire* input summary matrix!

    Args:
        M_Tqp ([type]): (steps,options,parameters) summary matrix to find the trajectory through
        beam_width ([type], optional): keep best beam_width solutions at each step T. Defaults to None.
        verb (int, optional): verbosity level for logging. Defaults to 0.

    Returns:
        tuple: Py_Tq,1,W_p - trajectory, 1, parameters
    """
    if beam_width is None:
        beam_width = M_Tqp.shape[1]
    Py_Tq = np.zeros(M_Tqp.shape[:2],
                     dtype=M_Tqp.dtype) / M_Tqp.shape[1]  # flat prior

    node2state_Tk = np.zeros((M_Tqp.shape[0], beam_width),
                             dtype=int)  # choosen state at each decision point
    node2parent_Tk = np.zeros((M_Tqp.shape[0], beam_width),
                              dtype=int)  # parent node for each decision point
    node2score_Tk = np.zeros(
        (M_Tqp.shape[0], beam_width),
        dtype=np.float32)  # score for each node at each decision point

    def backtrack(node2state_Tk, node2parentnode_Tk, M_Tqp):
        W_Tkp = np.zeros(
            node2state_Tk.shape + (M_Tqp.shape[-1], )
        )  # parameter settings along trajectory for each time point
        G_Tkp = np.zeros(
            node2state_Tk.shape + (M_Tqp.shape[-1], )
        )  # summary statistics along trajectory for each time point
        traj_Tk = np.zeros(node2state_Tk.shape,
                           dtype=int)  # node trajectory over time points
        node_k = list(
            range(node2state_Tk.shape[-1])
        )  # current state for each of the k trajectories we are backtracking
        for ti in range(node2state_Tk.shape[0] - 1, -1, -1):  # BWD in time
            # accumulate info for each state and backup
            for i, node in enumerate(node_k):
                q = int(node2state_Tk[
                    ti, node])  # map from beam-search index to true state
                traj_Tk[ti, i] = q  # record the trajectory information
                W_Tkp[ti, i, :] = M_Tqp[ti,
                                        q, :]  # accumulate parameter settings
                G_Tkp[ti, i, :] = M_Tqp[
                    ti,
                    q, :]  # accumulate summary statistics?? W and G are identical?
                node_k[i] = int(
                    node2parentnode_Tk[ti, node])  # backtrack to parent state
        return W_Tkp, G_Tkp, traj_Tk

    # summary statistics for the current decision point
    W_kp = np.zeros(
        (beam_width, M_Tqp.shape[2]))  # current parameter settings for node k
    G_kp = np.zeros(
        (beam_width,
         M_Tqp.shape[2]))  # accumulated score along the trajectory for node k
    for trli in range(M_Tqp.shape[0]):  # iterate over trials
        # compute the updated solution for all possible pairs, pre + current
        Wti_qp = M_Tqp[trli, :, :]
        Wti_kqp = W_kp[:, np.newaxis, :] + Wti_qp
        G_kqp = G_kp[:, np.newaxis, :] + Wti_qp

        # TODO[]: should this normalize w.r.t. Cxx?
        # score this set of solutions
        nrm = np.linalg.norm(Wti_kqp, axis=-1, keepdims=True)
        nrm[nrm == 0] = 1  # guard div-by-zero
        nWti_kqp = Wti_kqp / nrm  # equalize the norm, for scoring
        J_kq = np.einsum("kqp,kqp->kq", G_kqp, nWti_kqp)
        if verb > 1:
            print("{} J_kq - |W_kq| = {}".format(
                trli, np.sum(np.abs(J_kq - np.linalg.norm(Wti_kqp, axis=-1)))))
            print("{} Wti_kqp - G_kqp = {}".format(
                trli, np.sum(np.abs(Wti_kqp - G_kqp))))
            if verb > 1:
                print("\n{:3}) ".format('q\k'), end='')
                for j in range(J_kq.shape[0]):
                    print("  {:4d}".format(j), end='')
                for i in range(J_kq.shape[1]):
                    print("\n{:3d}) ".format(i), end='')
                    for j in range(J_kq.shape[0]):
                        print("  {:4.2f}".format(J_kq[j, i]), end='')
                print()

        # pick the best one for each q
        parent_q = np.zeros((J_kq.shape[1], ), dtype=int)
        state_q = np.zeros((J_kq.shape[1], ), dtype=int)
        score_q = np.zeros((J_kq.shape[1], ), dtype=np.float32)
        W_qp = np.zeros((J_kq.shape[1], M_Tqp.shape[-1]), dtype=M_Tqp.dtype)
        G_qp = np.zeros((J_kq.shape[1], M_Tqp.shape[-1]), dtype=M_Tqp.dtype)
        maxi_q = np.argmax(J_kq,
                           axis=0)  # best parent node each current option
        for qi in range(J_kq.shape[1]):
            maxki = np.argmax(J_kq[:, qi])  # best parent for this state
            # store parent info
            parent_q[qi] = maxki
            state_q[qi] = qi
            score_q[qi] = J_kq[maxki, qi]
            W_qp[qi, :] = Wti_kqp[maxki, qi, :]
            G_qp[qi, :] = G_kqp[maxki, qi, :]

        # and then the best k from q -- best subset (given current score)
        if W_kp.shape[0] == W_qp.shape[0]:
            node2parent_Tk[trli, :] = parent_q
            node2state_Tk[trli, :] = state_q
            node2score_Tk[trli, :] = score_q
            W_kp = W_qp.copy()
            G_kp = G_qp.copy()
        else:  # k < q
            maxi_q = np.argsort(score_q)  # ascend
            maxi_q = maxi_q[::-1][:W_kp.shape[0]]
            for i, mi in enumerate(maxi_q):
                node2parent_Tk[trli, i] = parent_q[mi]
                node2state_Tk[trli, i] = state_q[mi]
                node2score_Tk[trli, i] = score_q[mi]
                W_kp[i, :] = W_qp[mi, :]
                G_kp[i, :] = G_qp[mi, :]

        if verb > 0:
            print("{}) score= {}".format(trli, node2score_Tk[trli, :]))
            print("{} J_k - |W_k| = {}".format(
                trli,
                np.sum(
                    np.abs(node2score_Tk[trli, :] -
                           np.linalg.norm(W_kp, axis=-1)))))
            print("{} W_kp - G_kp = {}".format(trli,
                                               np.sum(np.abs(W_kp - G_kp))))

        # TODO[]: validate accumulated vs. backtracked computation of W and G here..
        if verb > 1:
            W_Tkp, G_Tkp, traj_Tk = backtrack(node2state_Tk[:trli + 1, :],
                                              node2parent_Tk[:trli + 1, :],
                                              M_Tqp[:trli + 1, ...])
            print("{}) W_kp - W_kp1 = {}".format(
                trli, np.max(np.abs(W_kp - np.sum(W_Tkp, 0)))))
            for k in range(traj_Tk.shape[-1]):
                print("{:3d}) {} = {:3.2f}".format(
                    k, traj_Tk[:, k], np.linalg.norm(np.sum(W_Tkp[:, k, :],
                                                            0))))

    # return the highest scoring trajectory return the 1st of the last set = highest scoring
    node = np.argmax(node2score_Tk[-1, :])
    W_p = W_kp[node, :]
    # extract the best scoring trajectory and fill in the Py_Tq list
    s_T = [None] * node2state_Tk.shape[0]
    for ti in range(node2state_Tk.shape[0] - 1, -1, -1):
        s_T[ti] = node2state_Tk[ti, node]
        Py_Tq[ti, s_T[ti]] = 1
        node = node2parent_Tk[ti, node]  # backtrack
    # print all the trajectories we've found
    if verb > 2:
        import matplotlib.pyplot as plt
        plt.figure()
        best = np.argsort(node2score_Tk[-1, :])[::-1][:6]
        for i, node in enumerate(best):
            mu = np.sum(W_Tkp[:, node, :], 0)
            print("{}) |W| = {:3.2f}  J = {:3.2f} {}".format(
                node, np.linalg.norm(mu), node2score_Tk[-1, node],
                traj_Tk[:, node]))
            plt.subplot((len(best)) // 2, 2, i + 1)
            plt.plot(W_Tkp[:, node, :].T)
            plt.plot(mu / W_Tkp.shape[1], 'k-')
            plt.title("{}) |W|={:3.2f} J={:3.2f} {}".format(
                node, np.linalg.norm(mu), node2score_Tk[-1, node],
                traj_Tk[:, node]))

    #W_p = W_p / node2score_Tk.shape[0]
    return Py_Tq, 1, W_p[np.newaxis, :]


def scoreOutput_temporal(F_MTSked,
                         Y_TSye,
                         dedup0=None,
                         R_mket=None,
                         offset=0,
                         train_idx=None,
                         test_idx=None,
                         fittype='svd',
                         softmaxscale=2,
                         beam_width=None,
                         nocontrol: bool = False,
                         verb=0):
    '''
    score each output given information on which stim-sequences corrospend to which inputs

    Args

      F_MTSked (nM,nTrl,nSamp,nfilt,nE,d): similarity score for each event type for each stimulus
      Y_TSye (nTrl,nSamp,nY,nE): Indicator for which events occured for which outputs
               nE=#event-types  nY=#possible-outputs  nEpoch=#stimulus events to process
      offset (int): A (set of) offsets to try when decoding.  Defaults to None.
      dedup0 (int): remove duplicate copies of output O, >0 remove the copy, <0 remove objID==0 (used when cross validating calibration data)

    Returns
      G_MTSy : (nM,nTrl,nSamp,nY): similarity score for each input epoch for each output
      W_Mykd : computed spatial filter for each model/output/component combination

    Copyright (c) MindAffect B.V. 2020
    '''
    from mindaffectBCI.decoder.scoreOutput import dedupY0
    from mindaffectBCI.decoder.analyse_datasets import get_train_test_indicators
    #assert offset==0
    if F_MTSked.size == 0:
        G_MTSy = np.zeros(F_MTSked.shape[:4] + (Y_TSye.shape[-2], ),
                          dtype=F_MTSked.dtype)
        return G_MTSy

    # ensure input size by padding with singlentons
    Y_TSye = Y_TSye.reshape(
        (1, ) * (4 - Y_TSye.ndim) + Y_TSye.shape)  # (nTrl,nSamp,nY,nE)
    F_MTSked = F_MTSked.reshape((1, ) * (4 - F_MTSked.ndim) +
                                F_MTSked.shape)  #(nM,nTrl,nSamp,nfilt,nE,d)

    if dedup0 is not None and dedup0 is not False:  # remove duplicate copies output=0
        Y_TSye = dedupY0(Y_TSye, zerodup=dedup0 > 0)
    # ensure Y_TSye has same type of F_mTSe
    Y_TSye = Y_TSye.astype(F_MTSked.dtype)

    if not offset == 0:  # shift Y to match desired offset
        tmp = Y_TSye
        Y_TSye = np.zeros(Y_TSye.shape, dtype=Y_TSye.dtype)
        if offset > 0:  # offset>0 -> shift Y FORWARDS in time, t_x=offset -> t_y=0
            Y_TSye[:, offset:, ...] = tmp[:, :-offset, ...]
        elif offset < 0:  # offset<0 -> shift Y BACKWARDS in time, t_x=0 -> t_y=offset
            Y_TSye[:, :offset, ...] = tmp[:, -offset:, ...]

    if nocontrol is not None and nocontrol is not False:
        oY_TSye = Y_TSye.copy()
        ncy = np.mean(Y_TSye, axis=2, keepdims=True)
        Y_TSye = np.concatenate((Y_TSye, ncy), axis=2)

    # compute the summary statistics we need
    # only for the training data
    train_ind, test_ind = get_train_test_indicators(F_MTSked[0, ...], Y_TSye,
                                                    train_idx, test_idx)
    if verb > 0:
        print("scoreOutput_temporal:: Trn={}  Tst={}".format(
            np.sum(train_ind), np.sum(test_ind)))
    Cyxr_MTykd = np.einsum("MTSked,TSye->MTykd",
                           F_MTSked[:, train_ind, ...],
                           Y_TSye[train_ind, ...],
                           dtype=F_MTSked.dtype,
                           casting='unsafe')
    Cyxr_MTykd = Cyxr_MTykd / Y_TSye.shape[1]

    # TODO[]: add a prior?
    # TODO[]: impose orthogonality over the ranks of the outputs?
    if fittype.lower() == 'pery':
        # compute the optimal spatial filter
        # accumulate over trials, N.B. this assumes the 'correct' model is fixed over trials!
        W_Mykd = np.sum(Cyxr_MTykd, axis=1, keepdims=False)

        # make unit norm
        W_Mykd = W_Mykd / Cyxr_MTykd.shape[1]
        #W_Mykd = W_Mykd / np.sqrt(np.sum(W_Mykd**2,-1,keepdims=True))

        # apply the spatial filter to all the data to get the output score estimate
        G_MTSye = np.einsum("MTSked,Mykd->MTSye",
                            F_MTSked,
                            W_Mykd,
                            dtype=F_MTSked.dtype,
                            casting='unsafe')
        G_MTSy = np.einsum("MTSye,TSye->MTSy", G_MTSye, Y_TSye)

    elif fittype.lower() == 'tgt0':
        # compute the optimal spatial filter
        # accumulate over trials, N.B. this assumes the 'correct' model is fixed over trials!
        W_Mkd = np.sum(Cyxr_MTykd[:, :, 0, :, :], axis=1, keepdims=False)

        # make unit norm
        W_Mkd = W_Mkd / Cyxr_MTykd.shape[1]
        #W_Mykd = W_Mykd / np.sqrt(np.sum(W_Mykd**2,-1,keepdims=True))

        # apply the spatial filter to all the data to get the output score estimate
        G_MTSe = np.einsum("MTSked,Mkd->MTSe",
                           F_MTSked,
                           W_Mkd,
                           dtype=F_MTSked.dtype,
                           casting='unsafe')
        G_MTSy = np.einsum("MTSe,TSye->MTSy", G_MTSe, Y_TSye)

        # BODGE: make the same shape at output
        W_Mykd = W_Mkd[:, :, np.newaxis, :]

    elif fittype.lower() == 'em':  # expectation maximization approach
        #Cyxr_MTykd[...,0,:,:]=0 # remove true-tgt info
        Py_MTy = np.ones(
            (F_MTSked.shape[0], Cyxr_MTykd.shape[1], Cyxr_MTykd.shape[-3]),
            dtype=F_MTSked.dtype) / Y_TSye.shape[-2]  # flat prior

        #Py_MTy[...] = 0
        Py_MTy[..., 0] = 1
        if verb > 1: print("-) Py={}".format(np.mean(Py_MTy, axis=(0, 1))))
        for iter in range(20):  # iteratively fit Py and W
            # M-step : compute the optimal spatial filter
            W_Mkd = np.einsum("MTykd, MTy -> Mkd", Cyxr_MTykd,
                              Py_MTy) / Cyxr_MTykd.shape[1]
            # E-step : compute update responsibilities
            # apply the spatial filter to get the output score estimate
            G_MTy = np.einsum("MTykd,Mkd->MTy", Cyxr_MTykd, W_Mkd)
            # soft-max to get responsibilities -> Py_MTy
            # TODO[]: calibration of the log-likelihood score?
            # TODO[]: make valid SSE computation by including the Cyy part?  (which can vary over Y's)
            f_MTy = G_MTy - np.max(G_MTy, axis=-1,
                                   keepdims=True)  # numerical guard
            f_MTy = np.exp(f_MTy * softmaxscale)
            Py_MTy = f_MTy / np.sum(f_MTy, axis=-1, keepdims=True)

            if verb > 1:
                print("{}) Py={}".format(iter, np.mean(Py_MTy, axis=(0, 1))))

        # apply the spatial filter to get the output score estimate
        G_MTSe = np.einsum("MTSked,Mkd->MTSe",
                           F_MTSked,
                           W_Mkd,
                           dtype=F_MTSked.dtype,
                           casting='unsafe')
        G_MTSy = np.einsum("MTSe,TSye->MTSy", G_MTSe, Y_TSye)
        # BODGE: make the same shape at output
        W_Mykd = W_Mkd[:, :, np.newaxis, :]

    elif fittype.lower() == 'svd':  # svd clustering approach
        #Cyxr_MTykd[...,0,:,:]=0 # remove true-tgt info
        W_Mkd = np.zeros(
            (Cyxr_MTykd.shape[0], Cyxr_MTykd.shape[3], Cyxr_MTykd.shape[4]),
            dtype=Cyxr_MTykd.dtype)
        for mi in range(Cyxr_MTykd.shape[0]):
            for ki in range(Cyxr_MTykd.shape[-2]):
                M_Tyd = Cyxr_MTykd[mi, :, :, ki, :]
                M_Ty_d = M_Tyd.reshape((-1, M_Tyd.shape[-1]))
                # decompose
                Py_Ty_k, S_k, W_kd = np.linalg.svd(M_Ty_d)
                # extract biggest component
                Py_Ty = Py_Ty_k[:, 0]
                sgn = np.sign(np.sum(Py_Ty))
                W_Mkd[mi, ki, :] = W_kd[0, :] * sgn

        # apply the spatial filter to get the output score estimate
        G_MTSe = np.einsum("MTSked,Mkd->MTSe",
                           F_MTSked,
                           W_Mkd,
                           dtype=F_MTSked.dtype,
                           casting='unsafe')
        G_MTSy = np.einsum("MTSe,TSye->MTSy", G_MTSe, Y_TSye)
        # BODGE: make the same shape at output
        W_Mykd = W_Mkd[:, :, np.newaxis, :]

    elif fittype.lower() == 'dp':
        W_Mkd = np.zeros(
            (Cyxr_MTykd.shape[0], Cyxr_MTykd.shape[3], Cyxr_MTykd.shape[4]),
            dtype=Cyxr_MTykd.dtype)
        for mi in range(Cyxr_MTykd.shape[0]):
            for ki in range(Cyxr_MTykd.shape[-2]):
                M_Tyd = Cyxr_MTykd[mi, :, :, ki, :]
                Py_Ty, S_k, W_nd = dp(M_Tyd, beam_width=beam_width, verb=verb)
                if verb > 0:
                    for ti in range(Py_Ty.shape[0]):
                        print("{})".format(ti), end='')
                        print(" ".join("{:2.0f}".format(i)
                                       for i in Py_Ty[ti, :]))
                W_Mkd[mi, ki, :] = W_nd[0, :]

        # apply the spatial filter to get the output score estimate
        G_MTSe = np.einsum("MTSked,Mkd->MTSe",
                           F_MTSked,
                           W_Mkd,
                           dtype=F_MTSked.dtype,
                           casting='unsafe')
        G_MTSy = np.einsum("MTSe,TSye->MTSy", G_MTSe, Y_TSye)
        # BODGE: make the same shape at output
        W_Mykd = W_Mkd[:, :, np.newaxis, :]

    return G_MTSy, W_Mykd


def scoreStimulusContFreq(X_TSfd,
                          W_mkd,
                          R_mket=None,
                          b_me=None,
                          f_mf=None,
                          offset=0):
    '''
    Apply spatio-temporal (possibly factored) model to data 

    Args:
      X_TSfd (ndarray): (nTrl,nSamp,nFreq,d) raw response for the current stimulus event
         OR    X_TEtd (nTrl,nEpoch,tau,d) pre-sliced raw data
      W_mkd (ndarray): (nModel,rank,d) spatial filters for each output
      R_mket (ndarray): (nModel,rank,nEvent,tau) responses for each stimulus event for each output
      b_me : (nModel,nEvent) bias for each stimulus type
      f_mf : (nModel,nFreq) spectral filter weighting
      offset (int): offset in X for applying W

    Returns
      F_mTSe (ndarray): (nModel,nTrl,nSamp,nEvent) similarity score for each input epoch for each output
    Copyright (c) MindAffect B.V. 2018
    '''
    # ensure all inputs have the right shape, by padding earlier dims
    X_TSfd = X_TSfd.reshape((1, ) * (4 - X_TSfd.ndim) + X_TSfd.shape)
    W_mkd = W_mkd.reshape(((1, ) * (3 - W_mkd.ndim)) + W_mkd.shape)
    R_mket = R_mket.reshape(((1, ) * (4 - R_mket.ndim)) + R_mket.shape)
    if f_mf is not None and f_mf.ndim < 2:
        f_mf = f_mf[np.newaxis, :]
    if not b_me is None and b_me.ndim < 2:
        b_me = b_me[np.newaxis, :]

    tau = W_mkd.shape[-2] if R_mket is None else R_mket.shape[
        -1]  # est response length
    if X_TSfd.shape[-3] < tau:  # X isn't big enough to apply... => zero score
        F_mTSe = np.zeros(
            (W_mkd.shape[0], X_TSfd.shape[0], X_TSfd.shape[1], W_mkd.shape[1]),
            dtype=X_TSfd.dtype)
        return F_mTSe

    if True:  # window and apply all at once
        X_TStfd = window_axis(X_TSfd, winsz=tau,
                              axis=-3)  # (nModel, nTrl, nSamp-tau, tau, k)
        if f_mf.ndim == 2:  # global spectral filter
            F_mTSe = np.einsum("mkd,TStfd,mket,mf->mTSe",
                               W_mkd,
                               X_TStfd,
                               R_mket,
                               f_mf,
                               optimize='optimal')
        else:  # per-rank spectral filter
            f_mkf = f_mf
            F_mTSe = np.einsum("mkd,TStfd,mket,mkf->mTSe",
                               W_mkd,
                               X_TStfd,
                               R_mket,
                               f_mkf,
                               optimize='optimal')
    else:  # spatial apply -> temporal apply
        # 1) apply the non-time dependant parts (contract X)
        if f_mf.ndim == 2:  # global spectral filter
            wXf_mTSk = np.einsum("mkd,TSfd,mf->mTSk",
                                 W_mkd,
                                 X_TSfd,
                                 f_mf,
                                 optimize='optimal')
        else:  # per-rank spectral filter
            wXf_mTSk = np.einsum("mkd,TSfd,mkf->mTSk",
                                 W_mkd,
                                 X_TSfd,
                                 f_mf,
                                 optimize='optimal')

        # 2) apply the temporal parts
        # window to apply the temporal aspect
        wXf_mTStk = window_axis(wXf_mTSk, winsz=tau,
                                axis=-2)  # (nModel, nTrl, nSamp-tau, tau, k)
        F_mTSe = np.einsum("mTStk,mket->mTSe",
                           wXf_mTStk,
                           R_mket,
                           optimize='optimal')

    # include the bias, for each stimulus type
    if not b_me is None:
        F_mTSe = F_mTSe + b_me[:, np.newaxis, np.newaxis, :]

    # shift for the offset and zero-pad to the input X size
    # N.B. as we are offsetting from X->Y we move in the **OPPOSITTE** direction to
    # how Y is shifted!
    Feoffset = -offset
    if Feoffset <= 0:
        tmp = F_mTSe[..., -Feoffset:, :]  # shift-back and shrink
        F_mTSe = np.zeros(F_mTSe.shape[:-2] + (X_TSfd.shape[-3], ) +
                          F_mTSe.shape[-1:],
                          dtype=F_mTSe.dtype)
        F_mTSe[..., :tmp.shape[-2], :] = tmp  # insert
    else:
        tmp = F_mTSe[..., :X_TSfd.shape[-3] - Feoffset, :]  # shrink
        F_mTSe = np.zeros(F_mTSe.shape[:-2] + (X_TSfd.shape[-3], ) +
                          F_mTSe.shape[-1:],
                          dtype=F_mTSe.dtype)
        F_mTSe[...,
               Feoffset:Feoffset + tmp.shape[-2], :] = tmp  # shift + insert

    return F_mTSe


def plot_Fe(F_mTSe, evtlabs=None):
    import matplotlib.pyplot as plt
    '''plot the stimulus score function'''
    if evtlabs is None:
        evtlabs = range(F_mTSe.shape[-1])
    #print("F_mTSe={}".format(F_mTSe.shape))
    if F_mTSe.ndim > 3:
        if F_mTSe.shape[0] > 1:
            print('Warning stripping model dimension!')
        F_mTSe = F_mTSe[0, ...]
    elif F_mTSe.ndim < 3:
        F_mTSe = F_mTSe[np.newaxis,
                        ...] if F_mTSe.ndim == 2 else F_mTSe[np.newaxis,
                                                             np.newaxis, ...]
    plt.clf()
    nPlts = min(25, F_mTSe.shape[0])
    if F_mTSe.shape[0] / 2 > nPlts:
        tis = np.linspace(0, F_mTSe.shape[0] / 2 - 1, nPlts, dtype=int)
    else:
        tis = np.arange(0, nPlts, dtype=int)
    ncols = int(np.ceil(np.sqrt(nPlts)))
    nrows = int(np.ceil(nPlts / ncols))
    axploti = ncols * (nrows - 1)
    linespacing = np.max(np.abs(F_mTSe))
    ax = plt.subplot(nrows, ncols, axploti + 1)
    for ci, ti in enumerate(tis):
        # make the axis
        if ci == axploti:  # common axis plot
            pl = ax
        else:  # normal plot
            pl = plt.subplot(nrows, ncols, ci + 1, sharex=ax,
                             sharey=ax)  # share limits
            plt.tick_params(labelbottom=False, labelleft=False)  # no labels
        pl.plot(F_mTSe[ti, :, :] +
                np.arange(F_mTSe.shape[-1])[np.newaxis, :] * linespacing)
        pl.set_title("{}".format(ti))
    pl.legend(evtlabs)
    plt.suptitle('F_mTSe')
    return plt.gca()


#@function
def testcase(
    nE=1,
    d=8,
    tau=10,
    nSamp=300,
    nTrl=30,
    nfilt=1,
    nM=20,
    noise2signal=4,
    nY=40,
    isi=2,
):
    import matplotlib.pyplot as plt
    from mindaffectBCI.decoder.utils import testSignal

    #   X = (nTrl, nEp, tau, d) [d x tau x nEpoch x nTrl ] raw response for the current stimulus event
    #            d=#electrodes tau=#response-samples  nEpoch=#stimulus events to process
    #    w = (nM, nfilt, d) spatial filters for each output
    #    r = (nM, nfilt, nE, tau) responses for each stimulus event for each output
    # more complex example with actual signal/noise
    irf = (1, 1, -1, -1, 0, 0, 0, 0, 0, 0)
    X_TSd, Y_TSye, st, A, R = testSignal(nTrl=nTrl,
                                         nSamp=nSamp,
                                         d=d,
                                         nE=nE,
                                         nY=nY,
                                         isi=isi,
                                         irf=irf,
                                         noise2signal=noise2signal,
                                         seed=0)
    Y_TSy = Y_TSye[..., 0]  # (nTrl,nSamp,nY)
    W = np.linalg.pinv(A).T[0, :]

    X_TStd = window_axis(X_TSd, winsz=tau, axis=-2)
    #W = np.random.randn(nM, nfilt, d).astype(np.float32)
    #R = np.random.randn(nM, nfilt, nE, tau).astype(np.float32)
    F_MTSe = scoreStimulus(X_TStd, W, R)
    print("ss X_TStd={} -> F_MTSe={}".format(X_TStd.shape, F_MTSe.shape))

    Wf = factored2full(W, R)
    Fef = scoreStimulus(X_TStd, Wf)
    print("Wf={} -> Fef={}".format(Wf.shape, Fef.shape))

    print("Fe-Fef={}".format(np.max(np.abs(F_MTSe - Fef).ravel())))

    F_MTse = scoreStimulusCont(X_TSd, W, R)
    print("ssCont X={} -> F={}".format(X_TSd.shape, F_MTse.shape))

    F_MTSked = scoreStimulus_temporal(X_TSd, None, R)
    print("ss_temporal X_TSd={} -> F_MTSked={}".format(X_TSd.shape,
                                                       F_MTSked.shape))
    G_MTSy, W_Mykd = scoreOutput_temporal(F_MTSked, Y_TSye, dedup0=True)
    print("G_MTSy = {}".format(G_MTSy.shape))
    print("W_Mykd = {}".format(W_Mykd.shape))

    # shuffle Y to make that it's not always 0 that's the best..
    Ytrn_TSye = Y_TSye.copy()
    #tmp = Ytrn_TSye[1,:,1,:].copy()
    #Ytrn_TSye[1,:,1,:] = Ytrn_TSye[1,:,0,:]
    #Ytrn_TSye[1,:,0,:] = tmp
    for ti in range(Ytrn_TSye.shape[0]):
        idx = ti % Ytrn_TSye.shape[2]  # swap tgt seq
        print("{}) {} -> {}".format(ti, 0, idx))
        # swap true and tgt
        tmp = Ytrn_TSye[ti, :, 0, :].copy()
        Ytrn_TSye[ti, :, 0, :] = Ytrn_TSye[ti, :, idx, :]
        Ytrn_TSye[ti, :, idx, :] = tmp

    retrain_idx = slice(0, 3)
    fittype = 'svd'
    G_MTSy, W_Mykd = scoreOutput_temporal(F_MTSked[:, retrain_idx, ...],
                                          Ytrn_TSye[retrain_idx, ...],
                                          dedup0=True,
                                          nocontrol=True,
                                          fittype=fittype,
                                          offset=-1)
    print("G_MTSy = {}".format(G_MTSy.shape))
    print("W_Mykd = {}".format(W_Mykd.shape))

    plt.figure()
    plt.subplot(211)
    plt.plot(W / np.sqrt(W.T @ W), '-', linewidth=3, label='true')
    plt.title("True")
    plt.plot(W_Mykd.squeeze().T / np.linalg.norm(W_Mykd), label=fittype)
    plt.title('{} estimated ({}trl)'.format(fittype, retrain_idx))
    plt.legend()
    plt.subplot(212)
    plt.imshow(np.sum(G_MTSy[0, ...], axis=-2), aspect='auto')
    plt.colorbar()
    plt.title('Output scores')
    #plt.show()

    fittype = 'dp'
    G_MTSy, W_Mykd = scoreOutput_temporal(F_MTSked[:, retrain_idx, ...],
                                          Ytrn_TSye[retrain_idx, ...],
                                          dedup0=True,
                                          fittype=fittype,
                                          nocontrol=True,
                                          beam_width=20,
                                          verb=0)
    print("G_MTSy = {}".format(G_MTSy.shape))
    print("W_Mykd = {}".format(W_Mykd.shape))

    plt.figure()
    plt.subplot(211)
    plt.plot(W / np.sqrt(W.T @ W), '-', linewidth=3, label='true')
    plt.title("True")
    plt.plot(W_Mykd.squeeze().T / np.linalg.norm(W_Mykd), label=fittype)
    plt.title('{} estimated ({}trl)'.format(fittype, retrain_idx))
    plt.legend()
    plt.subplot(212)
    plt.imshow(np.sum(G_MTSy[0, ...], axis=-2), aspect='auto')
    plt.colorbar()
    plt.title('Output scores')
    plt.show()

    quit()

    # try different offsets...
    W = np.random.randn(1, nfilt, d).astype(np.float32)
    R = np.random.randn(1, nfilt, 1, tau).astype(np.float32)
    offsets = list(range(-15, 15))
    Fes = []
    for i, offset in enumerate(offsets):
        Fe = scoreStimulusCont(X_TSd, W, R, offset=offset)
        Fes.append(Fe)
    Fes = np.concatenate(Fes, -1)  # (nM,nTr,nSamp,nOffset)
    plot_Fe(Fes[0, 0, ...], offsets)
    plt.show()


if __name__ == "__main__":
    testcase()
