<div align="center">
<h1> <a href="https://arxiv.org/abs/2603.09084">OmniEdit: A Training-free framework for Lip Synchronization and Audio-Visual Editing </a></h1>
</div>

## Lip Synchronization Results

<table>
<tr>
<th style="text-align:center;">Original Video</th>
<th style="text-align:center;">Lip synchronization</th>
</tr>

<tr>
<td align="center">
<video src="examples/video1.mp4" controls style="width:100%"></video>
<video src="examples/video2.mp4" controls style="width:100%"></video>
</td>

<td align="center">
<video src="examples/video1_lipsync.mp4" controls style="width:100%"></video>
<video src="examples/video2_lipsync.mp4" controls style="width:100%"></video>
</td>
</tr>
</table>


## Audio-Visual Editing Results

<table>
<tr>
<th style="text-align:center;">Original video</th>
<th style="text-align:center;">Audio-visual Editing</th>
</tr>

<tr>
<td align="center">
<video src="examples/young.mp4" controls style="width:100%"></video>
<video src="examples/man.mp4" controls style="width:100%"></video>
<video src="examples/ori.mp4" controls style="width:100%"></video>
<video src="examples/happy.mp4" controls style="width:100%"></video>
<video src="examples/laugh.mp4" controls style="width:100%"></video>
<video src="examples/race_car.mp4" controls style="width:100%"></video>
</td>

<td align="center">
<video src="examples/old.mp4" controls style="width:100%"></video>
<video src="examples/woman.mp4" controls style="width:100%"></video>
<video src="examples/trump.mp4" controls style="width:100%"></video>
<video src="examples/sad.mp4" controls style="width:100%"></video>
<video src="examples/cry.mp4" controls style="width:100%"></video>
<video src="examples/police_car.mp4" controls style="width:100%"></video>
</td>
</tr>
</table>

### Installation
```
conda create -n omniedit python=3.12
conda activate omniedit
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install

git clone https://github.com/Lightricks/LTX-2.git

modify LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py line 137:
FlashAttention3()(q, k, v, heads, mask)

cd LTX-2/packages/ltx-core
pip install -e .
cd ../ltx-pipelines
pip install -e .

pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

### Model Preparation

#### Lip synchronization model
| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| HuMo-17B      | [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-17B)   | 17B model
| HuMo-1.7B | [Huggingface](https://huggingface.co/bytedance-research/HuMo/tree/main/HuMo-1.7B) | 1.7B model
| Wan-2.1 | [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | VAE & Text encoder
| Whisper-large-v3 |      [Huggingface](https://huggingface.co/openai/whisper-large-v3)          | Audio encoder

Download models using hf:
``` sh
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
hf download bytedance-research/HuMo --local-dir ./HuMo
hf download openai/whisper-large-v3 --local-dir ./whisper-large-v3
```


#### Audio-visual editing model

| Models       | Download Link                                                                                                                                           |    Notes                      |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| LTX-2 | [Huggingface](https://huggingface.co/Lightricks/LTX-2) | LTX-2
| LTX-2.3 | [Huggingface](https://huggingface.co/Lightricks/LTX-2.3) | LTX-2.3
| Gemma 3| [Huggingface](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) | Gemma 3

Download models using hf:
``` sh
hf download Lightricks/LTX-2 --local-dir ./LTX-2
hf download Lightricks/LTX-2.3 --local-dir ./LTX-2.3
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./gemma-3-12b-it-qat-q4_0-unquantized
```

### Lip Synchronization
```
python omniedit.py --config configs/lipsync17b.yaml
```

### Audio-visual Editing
```
python omniedit.py --config configs/avedit.yaml
```



## Acknowledgements

Our work builds upon and is greatly inspired by several outstanding open-source projects, including [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FlowEdit](https://github.com/fallenshock/FlowEdit), [FlowAlign](https://github.com/FlowAlign/FlowAlign), [Humo](https://github.com/Phantom-video/HuMo), [LTX-2](https://github.com/Lightricks/LTX-2). We sincerely thank the authors and contributors of these projects for generously sharing their excellent codes and ideas.


## Citation

If you find this project useful for your research, please consider citing our [paper](https://arxiv.org/abs/2603.09084).

### BibTeX
```bibtex
@misc{lin2026omniedit,
      title={OmniEdit: A Training-free framework for Lip Synchronization and Audio-Visual Editing}, 
      author={Lixiang Lin and Siyuan Jin and Jinshan Zhang},
      year={2026},
      eprint={2603.09084},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.09084}, 
}
```