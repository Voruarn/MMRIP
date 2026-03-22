# MMRIP, ESWA 2025</a> </p>

- Paper: [Incorporating estimated depth maps and multi-modal pretraining to improve salient object detection in optical remote sensing images](https://www.sciencedirect.com/science/article/pii/S0957417425032397)


## Abstract

As a burgeoning theme in optical remote sensing image (ORSI) analysis, salient object detection (SOD) plays a vital role in traffic monitoring, agriculture, disaster management, and other fields. However, the existing ORSI-SOD methods are all single-modal (RGB images primarily), which suffer from performance drop when facing complex scenes (e.g., intricate backgrounds, low contrast scenes, and similar objects). To address this challenge, we introduce estimated depth map to complement RGB image in ORSI-SOD for the first time, which provides 3D geometric cues to improve detection accuracy in complex scenes, thus advancing ORSI-SOD from single-modal to multi-modal. Furthermore, we design a novel pretraining framework: multi-modal reconstructed image pretraining (MMRIP) to pretrain SOD model in multi-modal ORSI-SOD. MMRIP initially utilizes a masked autoencoder (MAE) to restore the masked RGB image; subsequently, it feeds the restored RGB image and clean depth map to the SOD model to generate the saliency map, which can help SOD model more effectively integrate cross modal information and extract better feature. Besides, we present a simple RGB-D SOD model, namely SimSOD, which is pretrained by MMRIP for ORSI-SOD. SimSOD has two major components: DFormer (encoder) and MLP head (decoder). Specifically, we first input RGB image and depth data into the encoder to generate four multi-scale features, then use the decoder to fuse these features and yield the prediction result. Without bells and whistles, our proposed method outperforms the state-of-the-art methods on three public ORSI-SOD datasets. The code can be accessed at: https://github.com/Voruarn/MMRIP.

```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@article{FU2026129624,
    title = {Incorporating estimated depth maps and multi-modal pretraining to improve salient object detection in optical remote sensing images},
    journal = {Expert Systems with Applications},
    volume = {298},
    pages = {129624},
    year = {2026},
    issn = {0957-4174},
    author = {Yuxiang Fu and Wei Fang}
}
```
