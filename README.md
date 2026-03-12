<div align="center">
<h1> <a href="https://arxiv.org/abs/2603.09084">OmniEdit: A Training-free framework for Lip Synchronization and Audio-Visual Editing </a></h1>
</div>

## Lip Synchronization Results


<table>
<tr>
<td width="50%">

### Input Video
---
https://github.com/user-attachments/assets/a4e18b53-961c-4e87-915b-eafe6a0f6e60

---
https://github.com/user-attachments/assets/d9b9be92-bd3d-49f9-8eba-83e597420c63

</td>

<td width="50%">

### Lip synchronization
--- 
https://github.com/user-attachments/assets/e65bb114-b728-45fa-9e4c-91d0cdb3b687

---
https://github.com/user-attachments/assets/4dde6006-e1d7-408e-8077-4facad45346c
</td>
</tr>
</table>

<!-- <table>
<tr>
<th style="text-align:center;">Original Video</th>
<th style="text-align:center;">Lip synchronization</th>
</tr>

<tr>
<td align="center">
<video src="https://github.com/user-attachments/assets/a4e18b53-961c-4e87-915b-eafe6a0f6e60" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/d9b9be92-bd3d-49f9-8eba-83e597420c63" controls style="width:100%"></video>
</td>

<td align="center">
<video src="https://github.com/user-attachments/assets/e65bb114-b728-45fa-9e4c-91d0cdb3b687" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/4dde6006-e1d7-408e-8077-4facad45346c" controls style="width:100%"></video>
</td>
</tr>
</table> -->


## Audio-Visual Editing Results

<table>
<tr>
<td width="50%">

### Input Video
---
https://github.com/user-attachments/assets/3494af3c-5161-4c40-a3c4-fb2651488ec6

---
https://github.com/user-attachments/assets/5740766c-e069-4b08-a80b-6de3b74aa36c

---
https://github.com/user-attachments/assets/ca2a17f2-51fe-4740-8e68-1f4836948ab1

---
https://github.com/user-attachments/assets/0c287953-19f9-48cd-bbb0-8e8db2b5c593

---
https://github.com/user-attachments/assets/fd71642f-6c07-4231-8d51-49b528706b23

---
https://github.com/user-attachments/assets/d7f8df48-9ec0-49b4-8cab-91d6f095ba09

</td>

<td width="50%">

### Audio-visual Editing
---
https://github.com/user-attachments/assets/1d59ff39-7dab-47d6-a6c0-9fcebfcde1c5

---
https://github.com/user-attachments/assets/a278a471-5f7f-4f25-8193-485a57e319ae

---
https://github.com/user-attachments/assets/8bebada6-5f05-447c-9a61-9cde663a12a9

---
https://github.com/user-attachments/assets/d67ce375-d287-4013-a6ac-4015dc396b74

---
https://github.com/user-attachments/assets/2afc567f-0c0d-4137-ad7a-5be67855fa47

---
https://github.com/user-attachments/assets/9ce011ad-b24e-4bbf-bf9f-bd748f8b2235
</td>
</tr>
</table>


<!-- <table>
<tr>
<th style="text-align:center;">Original video</th>
<th style="text-align:center;">Audio-visual Editing</th>
</tr>

<tr>
<td align="center">
<video src="https://github.com/user-attachments/assets/3494af3c-5161-4c40-a3c4-fb2651488ec6" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/5740766c-e069-4b08-a80b-6de3b74aa36c" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/ca2a17f2-51fe-4740-8e68-1f4836948ab1" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/0c287953-19f9-48cd-bbb0-8e8db2b5c593" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/fd71642f-6c07-4231-8d51-49b528706b23" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/d7f8df48-9ec0-49b4-8cab-91d6f095ba09" controls style="width:100%"></video>
</td>

<td align="center">
<video src="https://github.com/user-attachments/assets/1d59ff39-7dab-47d6-a6c0-9fcebfcde1c5" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/a278a471-5f7f-4f25-8193-485a57e319ae" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/8bebada6-5f05-447c-9a61-9cde663a12a9" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/d67ce375-d287-4013-a6ac-4015dc396b74" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/2afc567f-0c0d-4137-ad7a-5be67855fa47" controls style="width:100%"></video>
<video src="https://github.com/user-attachments/assets/9ce011ad-b24e-4bbf-bf9f-bd748f8b2235" controls style="width:100%"></video>
</td>
</tr>
</table> -->

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