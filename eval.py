import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from scipy.ndimage import generate_binary_structure
from skimage.filters import threshold_yen
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from modules_EPDiff import *
from diffusion import *
from utils_eval_brats import *

def main():
    with open("./eval.yml", "r") as file_object:
        conf = yaml.load(file_object, Loader=yaml.SafeLoader)
        if conf["seed"] is not None:
            torch.manual_seed(conf["seed"])

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        device = accelerator.device
        model = UNet(
            c_in=conf["channels"],
            c_out=conf["channels"],
            device=device,
            img_size=conf["size"],
        )
        ckpt = torch.load(conf["model"])

        model.load_state_dict(ckpt)
        diffusion = Diffusion(
            noise_steps=conf["noise_steps"],
            img_size=conf["size"],
            beta_start=conf["beta_start"],
            beta_end=conf["beta_end"],
            device=device,
        )
        dataloader = MRI_Volume(
            conf,
            hist=False,
        )

        model, dataloader = accelerator.prepare(model, dataloader)
        pbar = tqdm(dataloader)
        threshold_test = [
            round(x, 3)
            for x in np.arange(conf["thr_start"], conf["thr_end"], conf["thr_step"])
        ]

        dice_scores = {i: [] for i in threshold_test}
        dice_scores_mf = {i: [] for i in threshold_test}
        my_auprs = {i: [] for i in ["aupr no median", "aupr", "auroc"]}

        with torch.no_grad():
            my_volume = []
            my_labels = []
            for i, (image, label) in enumerate(pbar):
                if conf['dataset'] == "lits":
                    image = image.unsqueeze(1)
                    image = torch.permute(image, (0, 1, 3, 4, 2))
                image = (image * 2) - 1   # ???
                num_volumes = image.shape[0]
                num_slices = image.shape[4]

                if conf['dataset'] == "lits":
                    image = torch.permute(image, (0, 4, 1, 2, 3))
                    label = torch.permute(label, (0, 2, 3, 1))
                else:
                    image = torch.permute(image, (0, 4, 1, 2, 3))
                image = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
                split = torch.split(image, conf["size_splits"])
                dts_list = []
                k = 0
                for my_tensor in split:
                    dts = diffusion.normative_diffusion(
                        model,
                        my_tensor,
                        k,
                        conf["start"],
                        conf["stop"],
                        conf["pyramid"],
                        conf["discount"],
                        conf["visualization"],
                    ).to("cpu")
                    k = k + 1
                    dts_list.append(dts)

                dts_list = torch.cat(dts_list, dim=0)
                if conf["gmean"] == True:
                    aggregation = gmean(dts_list, dim=1)
                elif conf["gfuse"] == True:
                    gm = gmean(dts_list, dim=1)
                    tm = torch.mean(dts_list, dim=1)
                    aggregation = (gm+tm)/2
                else:
                    aggregation = torch.mean(dts_list, dim=1)

                aggregation = aggregation.view(
                    num_volumes,
                    num_slices,
                    aggregation.shape[1],
                    aggregation.shape[2],
                    aggregation.shape[3],
                )
                aggregation = torch.permute(aggregation, (0, 2, 3, 4, 1))
                aggregation = aggregation.to(device)
                aggregation, label = accelerator.gather_for_metrics(
                    (aggregation, label)
                )
                my_labels.append(label.type(torch.bool).to("cpu"))
                my_volume.append(aggregation.to("cpu"))

            if accelerator.is_main_process:
                my_volume = torch.cat(my_volume, dim=0)
                my_labels = torch.cat(my_labels, dim=0)
                if conf["max"] == True:
                    anomaly_map = torch.max(my_volume, dim=1)[0]
                elif conf["single"] == True:
                    anomaly_map = my_volume[:,0,:,:,:]
                elif conf["att"] == True:
                    maxv = torch.max(my_volume, dim=1)[0]
                    meanv = torch.mean(my_volume, dim=1)
                    # anomaly_map = (maxv + meanv) / 2
                    anomaly_map = 0.4 * maxv + 0.6 * meanv
                elif conf["att+"] == True:
                    maxv = torch.max(my_volume, dim=1)[0]
                    meanv = torch.mean(my_volume, dim=1)
                    fv = my_volume[:,0,:,:,:]
                    anomaly_map = conf["x"] * maxv + conf["n"] * meanv + conf["f"] * fv
                else:
                    anomaly_map = torch.mean(my_volume, dim=1)

                anomaly_map_mf = torch.clone(anomaly_map)

                anomaly_map_mf = median_filter_3D(
                    anomaly_map_mf, kernelsize=conf["kernel_size"]
                )

                my_labels = my_labels.contiguous()
                anomaly_map = norm_tensor(anomaly_map)
                anomaly_map_mf = norm_tensor(anomaly_map_mf)

                anomaly_map = anomaly_map.contiguous()
                anomaly_map_mf = anomaly_map_mf.contiguous()
                aupr = average_precision_score(my_labels.view(-1), anomaly_map.view(-1))
                my_auprs["aupr no median"].extend([aupr])
                aupr = average_precision_score(
                    my_labels.view(-1), anomaly_map_mf.view(-1)
                )
                my_auprs["aupr"].extend([aupr])

                auroc = roc_auc_score( my_labels.view(-1), anomaly_map_mf.view(-1))
                my_auprs["auroc"].extend([auroc])

                for key in dice_scores:
                    segmentation = torch.where(anomaly_map > key, 1.0, 0.0)
                    segmentation = segmentation.type(torch.bool)
                    segmentation_mf = torch.where(anomaly_map_mf > key, 1.0, 0.0)
                    segmentation_mf = segmentation_mf.type(torch.bool)
                    dice_scores[key].extend(
                        [float(x) for x in dice(segmentation, my_labels)]
                    )
                    dice_scores[key] = np.mean(np.asarray(dice_scores[key]))
                    dice_scores_mf[key].extend(
                        [float(x) for x in dice(segmentation_mf, my_labels)]
                    )
                    dice_scores_mf[key] = np.mean(np.asarray(dice_scores_mf[key]))

                yen_segmentation = torch.zeros_like(anomaly_map)
                struc = generate_binary_structure(conf["rank"], conf["connectivity"])

                for j, volume in enumerate(anomaly_map):
                    thr = threshold_yen(volume.numpy())
                    segmentation = torch.where(volume > thr, 1.0, 0.0)
                    yen_segmentation[j] = segmentation
                yen_segmentation = bin_dilation(yen_segmentation, struc)
                dice_scores["yen"] = []
                dice_scores["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores["yen"] = np.mean(np.asarray(dice_scores["yen"]))

                for j, volume in enumerate(anomaly_map_mf):
                    thr = threshold_yen(volume.numpy())
                    segmentation = torch.where(volume > thr, 1.0, 0.0)
                    yen_segmentation[j] = segmentation
                yen_segmentation = bin_dilation(yen_segmentation, struc)
                dice_scores_mf["yen"] = []
                dice_scores_mf["yen"].extend(
                    [float(x) for x in dice(yen_segmentation, my_labels)]
                )
                dice_scores_mf["yen"] = np.mean(np.asarray(dice_scores_mf["yen"]))

                dice_scores["AUPRC"] = my_auprs["aupr no median"][0]
                dice_scores_mf["AUPRC"] = my_auprs["aupr"][0]
                df = pd.DataFrame.from_dict(
                    dice_scores, orient="index", columns=["value"]
                )
                df.index.name = "thr"
                df_mf = pd.DataFrame.from_dict(
                    dice_scores_mf, orient="index", columns=["value"]
                )
                df_mf.index.name = "thr"
                df.to_csv(conf["output"])
                df_mf.to_csv(conf["output_mf"])


if __name__ == "__main__":
    main()
