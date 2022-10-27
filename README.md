# Directory overview
    .
    ├── configs  # I use Hydra for command line arguments/hyperparameter tracking.
        # This is just a directory of the default configurations. I tried to document
        # some of the important parameters and their supported options inside the actual files.
    ├── diffusion_model  # Code for the diffusion model
    │   ├── StructuredPrediction
    │   │   ├── FeaturePreprocessor.py  # For preprocessing the input tokens,
                # i.e. projecting to Transformer dimensionality
    │   │   ├── SeqMaskPreprocessor.py  # For preprocessing the input sequences,
                # i.e. adding time conditioning, concatenating along seq/feat dim, creating masks
    │   │   └── Transformer.py  # The actual Transformer (and it's variants)
    │   ├── Backbones.py  # For extracting predictions & features from a pretrained backbone (or noisy GT)
    │   ├── PredictionHead.py  # Processes the output of the Transformer,
            # projects to data dim. and computes loss/output depending on train/inference
    │   ├── StructuredDiffusionModel.py  # The diffusion model, combines all of the above
    │   └── model_utils.py  # Various model utils, mostly various time-conditioned MLPs and
            # code for extracting sinusoidal embeddings
    ├── utils
    │   ├── cluster  # Various utils I wrote to interact with the cluster. Left the ones here that might be useful 
    │   │   ├── config_tune.yaml  # This and tune.py were for launching hyperparameter sweeps (random searches) 
    │   │   ├── tune.py
    │   │   ├── parent.py  # For selecting the hyperparameters of typical bolt jobs (grid search)
    │   │   ├── requirements.txt  # Project dependencies
    │   ├── data
    │   │   ├── BBNormalizer.py  # For normalizing bounding boxes to [-1, 1] 
    │   │   ├── BBox.py  # Bounding box class implementation. Helpers for converting between formats,
                # storing data related to a set of BBs, masking and unmasking  
    │   │   ├── coco_dataset.py  # Dataloader for COCO & stackable blocks datasets 
    │   ├── diffusion
            # This structure is kind of an artifact from when I thought our categorical
            # diffusion would be very different from typical diffusion. 
            # categorical_diffusion/ is kind of useless (as we used Analog Bits for classification
            # which is basically continuous diffusion). The root files (DiffusionLosses.py, etc.)
            # just aggregate the gaussian_diffusion/ and categorical_diffusion/ classes together
            # (though again, categorical_diffusion/ does nothing at this point)
    │   │   ├── categorical_diffusion
    │   │   │   ├── CategoricalDiffusionLosses.py
    │   │   │   ├── ForwardCategoricalDiffusion.py
    │   │   ├── gaussian_diffusion
    │   │   │   ├── ForwardGaussianDiffusion.py
    │   │   │   ├── GaussianDiffusionLosses.py
    │   │   │   └── ReverseGaussianDiffusion.py
    │   │   ├── DiffusionLosses.py
    │   │   ├── ForwardDiffusion.py
    │   │   └── ReverseDiffusion.py
    │   ├── evaluation
    │   │   ├── HistogramEvaluation.py  # Code for computing & plotting the histograms we looked at
                # at the start (like MMD). The code for visualizing samples is also here.
    │   │   ├── NLLEvaluation.py  # For computing NLL scores
    │   │   └── RMSEEvaluation.py  # For computing RMSE
    │   ├── callbacks.py  # This one got a little messy. Some of it is for calling the callbacks in evaluation/,
            # but some evaluation code ended up in here (like computing mAP, validity scores for stackable blocks).
    │   ├── class_formatting.py  # Just a couple simple functions for converting between
            # integers & bits & one-hot encodings
    │   ├── matcher.py  # DETR matching code
    │   └── setup_utils.py  # For setting up the dataloaders and models and stuff at the start of training 
    ├── COCO_EDA.ipynb  # The EDA we did at the start
    ├── app.py  # My code for visualizing jobs from Bolt
    ├── sample.py  # For sampling from a model with different strategies & number of steps
    ├── setup.sh  # run source setup.sh to setup the project on an Apple cluster
    ├── train.py
    
# Visualizing results
The quip doc has some plots for aggregating results from Bolt jobs. To run it, use `python app.py {Bolt job IDs}`. I usually just run this locally, though I know there is a way to connect to the port if you want to run it on a compute node. To view the results, just open the link it gives in a browser (default for me is http://127.0.0.1:8050). If you try to view more than one job at a time, it'll tell you the port is busy, so just add something like `python app.py {Bolt job IDs} --port=8060`. Also, you can tell it to redownload metrics with the flag:  `python app.py {Bolt job IDs} --redownload`. If the list of bolt jobs contains more than 50 child tasks, it will download them in batches. I never got around to aggregating the results from each batch, so you should rerun the command after you've downloaded results if there are more than 50 tasks :). 

# Running jobs <br>
Run `source setup.sh` to download all the dependencies and setup the environment. Then `python train.py {arguments}` :) (see configs/ for options). One nuance with Hydra is that you don't need the double dash like with argparse (see below).

Some arguments/setups that gets used/I was tweaking a lot:

**Noisy GT conditioning** <br>
Set these arguments (i.e. add these lines without the comments after train.py): <br>
model/backbone=gt  # Select the backbone to be the GT <br>
model.backbone.static_noise=True  # Sample noise at the start of training and keep it static <br>
model.backbone.num_timesteps=4000  # The number of diffusion steps for the process used to noise GT <br>
model.backbone.timestep=0.025  # The timestep to noise the GT to (in this example and what I always used 4000 * 0.025 = 100) <br>
model/structured/conditioning=bb_preds  # Use the BB predictions from bbone in conditioning <br>
model.structured.conditioning.concat_method=feats OR seq  # Concat conditioning along seq or feat. dim (only really works for encoder arch)

**Diffusion params** <br>
See the diffusion section in configs/config.yaml

**Architecture** <br>
model/structured=transformer-encoder OR transformer-decoder OR transformer

**Conditioning on image feats** <br>
model/structured/conditioning=both OR feats  # Both for both features & BB preds, feats for just CNN feats <br>
model.structured.conditioning.method=seq OR roi OR both  # Seq for DETR-style conditioning, roi for ROI-pooled feats, both for both (see Quip doc) <br>
model/backbone=detectron2 OR gt  # detectron2 if you want to use Faster-RCNN BBox predictions (and CNN feats), gt if you want noisy GT bboxes and Faster-RCNN image feats







