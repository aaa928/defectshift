"""TabNet (Arik & Pfister, AAAI 2021) training and prediction worker. Runs in the TensorFlow 1 environment (Python 3.8,
nvidia-tensorflow 1.15; environment/requirements-tabnet-tf1-lock.txt). defectshift/models/tabnet.py in the main environment passes
the preprocessed arrays (npz) and the job settings (json) and calls this script as a subprocess. It does not import defectshift
(different environment).

Official code: tabnet_model.TabNet of google-research/tabnet (commit 0a1b0fe, Apache-2.0) is imported unmodified and its encoder
and regress are used. The training operations are those of the official experiment_covertype.py: loss = task loss (MSE for
regression) + lambda_sparse x total_entropy, Adam + tf.train.exponential_decay, gradients clipped with
clip_by_value(+-grad_thresh) and applied under the UPDATE_OPS dependency (batch-normalization moving statistics). Differences from
the official example (see also docs/reproducing.md, "NODE and TabNet implementation notes"):
- Input columns follow the feature-column scheme of data_helper_covertype.get_columns (numeric_column; 1-dimensional
  embedding_column). The categories are already integer ids 0..K-1 (K = dataset-fixed cardinality), so they enter through
  categorical_column_with_identity instead of hash buckets, the same mapping without collisions.
- Batches: as in the official input_fn (shuffle, repeat, batch) the batches cross epoch boundaries and no row is dropped. Each
  epoch appends a permutation of the training rows, only full batch_size batches are cut, and the remaining rows are carried over
  to the next epoch (ghost batch norm needs a batch that is a multiple of virtual_batch_size).
- Training loop (common protocol rule): at the end of each epoch the MSE on the early-stopping rows (standardized y) is computed;
  the weights are saved on improvement, training stops after patience + 1 consecutive epochs without improvement, and the best
  epoch is restored. The official code runs a fixed number of steps (MAX_STEPS).

Usage: python defectshift/models/tabnet_tf1_worker.py <job.json>
job.json: data (npz: num_fit, cat_fit, y_fit, num_es, cat_es, y_es, num_<k>, cat_<k>), out (npz), pred_sets ([k...]), params (dict),
          cat_cardinalities, seed, max_epochs, patience, tabnet_dir, ckpt_prefix (when saving; otherwise temporary), restore (if
          present, restore and predict without training), device ("gpu"|"cpu")
out.npz: pred_<k> (standardized scale). out.json: record (best epoch, number of epochs, best early-stopping MSE, times, actual
batch size, ...).
"""
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")     # NVIDIA TF 1.15: deterministic GPU operations (where available)

import numpy as np  # noqa: E402

EVAL_BATCH = 4096


def build(tf, tabnet_model, n_num, cards, p):
    """The official TabNet graph. Returns (input placeholder dict, y placeholder, training op, loss, evaluation predictions)."""
    feats, columns = {}, []
    if n_num:
        feats["num"] = tf.placeholder(tf.float32, [None, n_num], name="num")
        columns.append(tf.feature_column.numeric_column("num", shape=(n_num,)))
    for j, k_card in enumerate(cards):
        key = f"cat{j:03d}"
        feats[key] = tf.placeholder(tf.int64, [None], name=key)
        columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(key, int(k_card)), dimension=1))
    y_ph = tf.placeholder(tf.float32, [None], name="y")
    model = tabnet_model.TabNet(
        columns=columns, num_features=n_num + len(cards), feature_dim=p["n_d"] + p["n_a"], output_dim=p["n_d"],
        num_decision_steps=p["n_steps"], relaxation_factor=p["gamma"], batch_momentum=p["batch_momentum"],
        virtual_batch_size=p["virtual_batch_size"], num_classes=1)
    enc_tr, total_entropy = model.encoder(feats, reuse=False, is_training=True)
    pred_tr = model.regress(enc_tr, reuse=False)
    enc_ev, _ = model.encoder(feats, reuse=True, is_training=False)
    pred_ev = model.regress(enc_ev, reuse=True)
    mse = tf.losses.mean_squared_error(tf.expand_dims(y_ph, 1), pred_tr, reduction=tf.losses.Reduction.MEAN)
    loss = mse + p["lambda_sparse"] * total_entropy
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(p["learning_rate"], global_step=global_step, decay_steps=p["decay_steps"],
                                    decay_rate=p["decay_rate"])
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = opt.compute_gradients(loss)
        g = float(p["grad_thresh"])
        train_op = opt.apply_gradients([(tf.clip_by_value(gr, -g, g), v) for gr, v in gvs], global_step=global_step)
    return feats, y_ph, train_op, loss, tf.squeeze(pred_ev, 1)


def feed(feats, num, cat, idx, y_ph=None, y=None):
    d = {}
    if "num" in feats:
        d[feats["num"]] = num[idx]
    for j in range(cat.shape[1]):
        d[feats[f"cat{j:03d}"]] = cat[idx, j]
    if y_ph is not None:
        d[y_ph] = y[idx]
    return d


def predict(sess, feats, pred_ev, num, cat):
    n = len(num)     # arrives with shape (n, 0) even without numerical columns
    out =[sess.run(pred_ev, feed_dict=feed(feats, num, cat, np.arange(s, min(s + EVAL_BATCH, n))))
           for s in range(0, n, EVAL_BATCH)]
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def main():
    job = json.load(open(sys.argv[1]))
    if job.get("device") == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    sys.path.insert(0, job["tabnet_dir"])
    import tabnet_model   # the official code as is

    z = np.load(job["data"])
    p, seed = job["params"], int(job["seed"])
    cards = list(job["cat_cardinalities"])
    n_num = int(z["num_fit"].shape[1])
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    feats, y_ph, train_op, loss_op, pred_ev = build(tf, tabnet_model, n_num, cards, p)
    saver = tf.train.Saver(max_to_keep=1)
    info = dict(tf_version=tf.__version__, params=p, seed=seed)
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True    # querying the GPU before the first session (tf.test.is_gpu_available) would claim memory with the default setting, so it is not done
    work = None
    with tf.Session(config=cfg) as sess:
        info["gpu"] = any(d.device_type == "GPU" for d in sess.list_devices())
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
        if job.get("restore"):
            saver.restore(sess, job["restore"])
            info.update(restored_from=job["restore"])
        else:
            work = tempfile.mkdtemp(prefix="tabnet_", dir=job.get("tmp_dir"))
            ckpt = job.get("ckpt_prefix") or os.path.join(work, "best")
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            num_fit, cat_fit, y_fit = z["num_fit"], z["cat_fit"], z["y_fit"].astype(np.float32)
            num_es, cat_es, y_es = z["num_es"], z["cat_es"], z["y_es"].astype(np.float64)
            n_fit, bs = len(y_fit), int(p["batch_size"])
            if bs > n_fit or bs % int(p["virtual_batch_size"]):
                raise ValueError(f"batch_size {bs} exceeds the {n_fit} training rows or is not a multiple of virtual_batch_size {p['virtual_batch_size']}")
            rng = np.random.RandomState(seed)
            best, best_epoch, epoch, t0, hist, steps = float("inf"), 0, 0, time.time(), [], 0
            queue = np.zeros(0, dtype=np.int64)
            for epoch in range(1, int(job["max_epochs"]) + 1):
                queue = np.concatenate([queue, rng.permutation(n_fit)])     # repeat -> batch (remaining rows carried over to the next epoch)
                n_b = len(queue) // bs
                for k in range(n_b):
                    sess.run(train_op, feed_dict=feed(feats, num_fit, cat_fit, queue[k * bs:(k + 1) * bs], y_ph, y_fit))
                queue, steps = queue[n_b * bs:], steps + n_b
                mse = float(np.mean((predict(sess, feats, pred_ev, num_es, cat_es) - y_es) ** 2))
                hist.append(mse)
                if not np.isfinite(mse):
                    raise FloatingPointError(f"early-stopping MSE is not finite at epoch {epoch}")
                if mse < best:
                    best, best_epoch = mse, epoch
                    saver.save(sess, ckpt, write_meta_graph=False, write_state=False)
                if epoch - best_epoch > int(job["patience"]):
                    break
            saver.restore(sess, ckpt)
            info.update(best_epoch=best_epoch, n_epochs=epoch, best_val_mse_std=best, train_sec=time.time() - t0,
                        n_steps=steps, batch_size=bs, val_mse_hist=hist,
                        ckpt_prefix=job.get("ckpt_prefix"))
        out, pred_sec = {}, {}
        for k in job["pred_sets"]:
            t1 = time.time()
            out[f"pred_{k}"] = predict(sess, feats, pred_ev, z[f"num_{k}"], z[f"cat_{k}"])
            pred_sec[k] = time.time() - t1
        info["predict_sec"] = pred_sec
    if work:
        shutil.rmtree(work, ignore_errors=True)
    np.savez(job["out"], **out)
    with open(os.path.splitext(job["out"])[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
