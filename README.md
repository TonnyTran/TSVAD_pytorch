*TS-VAD*

**Usage**
- 两个pretrain模型放在pretrain文件夹里 
  - Wavlm-base+: https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link
  - ecapa-tdnn: https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/exps/pretrain.model. (记得改名字成ecapa-tdnn.model）
- 先run prepare_data.py 生成 1. target_speech_data 2. training and eval list
- 再bash run.sh, 更改其中的路径
- Best Result: DER = 13.61

**Update**
- 相较上一版的更新
- 1. 短的speech变成slience
- 2. 短的间隔变成speech （这俩可以提升0.几个点）
- 3. 加入smooth最后的label (提升0.几个点)
- 4. 模型换成cnceleb上的resnet （不清楚作用，目测不大）
- 5. speech encoder变成12层的, 后面两个trans变成2/3层（应该是有用的，多有用没看）
- 6. warm up 训练20epoch， speech encoder冻结，之后一起train (不这么设定似乎不能train)
- 7. warm up之后， ts和rs都加噪 （不清楚有没有用）
- 8. 测试加入间隔 （dis = self.rs_len在dataloader里， 可以设定成25*2，即为2s间隔），能提升1个点左右
- 9. 直接用dscore算der

**To do**
- 和Speaker diraization模型连一起看看最后结果
- 模型我乱搭的 看看怎么改
- 加载pretrain的模型的代码写的稀碎，可以改一下

**Conclusion**
- 正常的speaker_ids效果好一些
