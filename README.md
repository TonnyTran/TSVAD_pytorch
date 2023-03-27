*TS-VAD*

**Usage**
- 两个pretrain模型放在pretrain文件夹里 
  - Wavlm-base+: https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link
  - ecapa-tdnn: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/exps/pretrain.model. (记得改名字成ecapa-tdnn.model）
- 先run prepare_data.py 生成 1. target_speech_data 2. training and eval list
- 再bash run.sh, 更改其中的路径
- Best Result: DER = 13.61

**To do**
- 和Speaker diraization模型连一起看看最后结果
- 模型我乱搭的 看看怎么改
- 加载pretrain的模型的代码写的稀碎，可以改一下

**Conclusion**
- 正常的speaker_ids效果好一些