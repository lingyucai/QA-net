#train_entry(config)
'''
config.py
处理数据
os.makedirs 创建根目录,如果不存在则创建
flags.DEFINE_string 等DEFINE函数，用以定义变量
config 定义了程序需要的所有全局参数变量
with open ( , "rb") 不会将"\rn"替换成"\n"  "r"会将"\rn"替换成"\m"
DepthwiseSaparableConv(96, 96, 7)
depthwise = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=7, groups=96, padding=3, bias=False)  输出大小为(in-kernel+2*pad)/stride+1)，因此不变
pointwise = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=1, padding=0, bias=True)
Initialized_Conv1d 对conv1d进行重写 kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False
这样赋值是为了使conv1d适用于FeedForwardlayer
并且对relu做了分支，若relu=True，weight进行kaiming正则化，relu=False，weight进行xavier_uniform
kaiming使得每一卷积层的输出的方差都为1, xavier_uniform为了保证输入输出的方差不变,在tanh激活函数上有很好的效果，但不适用于ReLU激活函数. 它适用激活函数线性
nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)]) 对应convs的layernorm层
torch.empty(96,1)定义形状
nn.init.xavier_uniform_(w4C)初始化张量
self.w4C = nn.Parameter(w4C)将张量值赋给模型参数，作为nn的参数。
optim.lr_scheduler.LambdaLR 用初始lr乘上一个给定的函数来赋值learning rate
'''
'''
Ccid[16, 400, 16]
Qcid[16, 50, 16]
Cwid[16, 400]
Qwid[16, 50]
embedding的index 
maskC = (torch.zeros_like(Cwid) != Cwid).float()
maskC 标记Cwid不为0的那些项，相应位置标记为1
将index进行embedding获得字与词的向量表示
ch_emb[16, 400, 16, 64] -permute-> [16, 64, 400, 16]
F.dropout(ch_emb, p=dropout_char, training=self.training) p=0.05 在前向传播中以0.05概率使某些神经元失效
conv2d(64, 96, kernel_size=(1, 5), stride=(1,1)) -> ch_emb[16, 96, 400, 12] 
前两个参数决定了输入输出的通道数，输出的feature map大小由input的大小和kernel的大小决定 12=16-5+1
ch_emb, _ = torch.max(ch_emb, dim=3) maxpooling 略去index -> ch_emb [16,96,400,1] 将word_len通过maxpooling转为1
ch_emb = ch_emb.squeeze() -> ch_emb[16,96,400] 降维
wd_emb[16, 400, 300] 同样先dropout
wd_emb.transpose(1,2) -> [16, 300, 400]
emb = torch.cat([ch_emb, wd_emb], dim=1 emb->[16, 396, 400]
emb = self.conv1(emb) Conv1d(396, 96, kernel_size=1,stride=(1,)) ->[16, 96, 400]
Highway
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
D = 92


class Highway(nn.Module):
    def __init__(self, layer_num: int, size=D):
        super().__init__()
        self.n = layer_num #2
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for_in range(self.n)])
        #输入和输出的维度不改变，都是96
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])
        # relu都为False
    def forward(self, x):
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            #Highway 的公式 y = H(x,Wh) * T(x,Wt) + x * (1-T) 这里非线性部分H用的是一个linear function（怀疑写错，应该加上relu） 门用的是带sigmoid的linear function
            nonlinear = F.dropout(nonlinear, p=dropout, trainging=self.training)
            x = gate * nonlinear + (1 - gate) * x m
            #Highway 要求 gate 和 nonlinear都是同样维度的， 最后进行点乘后不改变向量的维度。
        return x
    '''
    emb -> [16, 96, 400]
    blks = 1 blocks的数目
    conv_num = 4 conv层的数目
    ch_num = 1 
    k = 1
    '''
    Ce = self.emb_enc(C, maskC, 1, 1)
def PosENcoer(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1,2)
    # x[16, 96, 400] -> [16, 400, 96]
    length = x.size()[1] #length:400
    channels = x.size()[2] #channels:96
    signal = get_timing_signal(400, 96, 1.0, 1.0e4)
    return (x + signal.cude()).transpose(1,2)
'''
position [400]
num_timescales:48
log_timescales_increment = log(10000/1)/(48-1)=0.20
inv_timescales = 1*exp(torch.arange(48) * -0.20) -> [48]
scaled_time = [400,1] * [1, 48] = [400, 48]
signal = torch.cat([400,48], [400,48], dim=1)=[400,96]
nn.Zeropad2d((0,0,0,0))
signal -pad->[400,96]
signal = signal.view(1, 400, 96)
PosEncoder return (x + signal.cuda()).transpose(1,2)-> [16,96,400] 不同batch的相应的position值是相同的
'''
class EncoderBlock(nn.Module):
    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num+1)*blks #= (4+1)*1 = 5
        out = PosEncoder(x) #-> [16, 96, 400]不改变维度
        for i,conv in enumerate(self.convs):
            res = out #residual 为layernorm前的输入
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            #If a single integer is used, it is treated as a singleton list, and this module will normalize over the
            # last dimension which is expected to be of that specific size.
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=True) #当i是2的倍数的时候，就dropout
            out = conv(out)#DSC ->[16, 96, 400]
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)# ->[16, 96, 400]
            # dropout 由0.02->0.1 以0.02为间隔增大
            l += 1
        res = out #完成了conv的repeat，进入self-attention，同样构建residual
        out = self.norm_1(out.transpose(1,2)).transpose(1,2) #对96进行norm layernorm
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)# 没有改变维度
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)#layer dropout 就是 residual block 以0.1为dropout
        l += 1
        res = out # [16, 96, 400]

        out = self.nort_2(out.transpose(1,2).transpose(1,2))
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)#conv1d(96,96,1,1)
        out = self.FFN_2(out)#conv1d(96,96,1,1)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)#dropout=0.12 直接返回residual的概率越来越大
        return out

    def layer_dropout(self, inputs, residual, dropout=0.02):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout #利用均匀分布随机生成0-1之间的数，以概率为2%来选择是否只返回惨差。
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual #在返回y前又加上了dropout
        else:
            return inputs + residual #如果不能训练，直接返回inputs + residual
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        #nn.Conv1d(96, 96, kernel_size=7, padding=3, groups=96, stride=1, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias)
        #nn.Conv1d(96, 96, kernel_size=1, stride=1)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))
    #depthwise改变的是last dimension，pointwise改变的是 last second dimention
    #k是奇数就成立
def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return  inputs + (-1e30) * (1 - mask)#为0的点使其无穷小

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem_conv = Initialized_Conv1d(D, D*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries
        memory = self.mem_conv(memory) # -> [16, 192, 400] 分别是q,k的linear层，将两个linear层进行了合并
        query = self.query_conv(queries) # ->[16, 96, 400]
        memory = memory.transpose(1, 2) # -> [16, 400, 192]
        query = query.transpose(1,2) # ->[16, 400, 96]
        Q = self.split_last_dim(query, 1) # multihead = 1 ->[16, 1, 400, 96] 将最后一维用第二维分割
        K, V = [self.split_last_dim(tensor, Nh) for tensor in torch.split(memory, D, dim=2)] #[16, 400 192] -> 2 * [16, 400, 96]
        # torch.split(memory, 96, dim=2) [16, 400, 192] 在dim=2上切分，且切分的单位是96
        key_depth_per_head = D // Nh # dk->96   Q和K的最后一维 scaling factor
        Q *= key_depth_per_head**-0.5 # scale
        x = self.dot_product_attention(Q, K, V, mask=mask) # [16, 1, 400, 96] 对最后两维进行计算，其实是对所有的multi-head进行了计算
        # dot_product_attention 将Q，K，V三个相同维数的矩阵进行计算，得到与输入维度相同的输出
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2) # [16, 400, 1, 96] -> [16, 96, 400]

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        # q:[16, 1, 400, 96] k:[16, 1, 400, 96] v:[16, 1, 400, 96] dot_product_attention 计算的是最后两维，处理后维度都相同
        logits = torch.matmul(q, k.permute(0,1,3,2)) # matmul([16, 1, 400, 96], [16, 1, 96, 400] -> [16, 1, 400, 400] Q KT V
        if bias:
            logits += self.bias
        if mask is not None:#mask用来标记每个句子在没有补零前原始长度
            shapes = [x if x != None else -1 for x in list(logits.size())] # [16, 1, 400, 400]
            mask = mask.view(shapes[0], 1, 1, shapes[-1]) # mask[16, 400] -> [16, 1, 1, 400] 在mask前需要对head所在维和q所在的维进行平铺
            logits = mask_logits(logits, mask) # mask_logits 进行Mask(opt.)
        weights = F.softmax(logits, dim=-1) #[16, 1, 400, 400]不改变shape 在word_size上做softmax
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v) # [16, 1, 400, 96]

    def split_last_dim(self, x, n):
        old_shaple = list(x.size()) #[16, 400, 96]
        last = old_shape[-1] #96
        new_shape = old_shape[:-1] + [n] + [last // n if last else None] # list相加成为新的list [16, 400, 1, 96]
        ret = x.view(new_shape) # -> [16, 400, 1, 96]
        return ret.permute(0, 2, 1, 3)# -> [16, 1, 400, 96]

    def combine_last_two_dim(self, x):
        # Reshape x so that the last two dimension become one
        old_shape = list(x.size()) # [16, 400, 1, 96]
        a, b = old_shape[-2:] # a: 1 b: 96
        new_shape = old_shape[:-2] + [a * b if a and b else None] # [16, 400, 1*96]
        ret = x.contiguous().view(new_shape)#contiguous把大矩阵的不同元素内存空间放在一起，利用view压缩时需要用到
        return ret
'''
C [16, 96, 400] Q[16, 96, 50]
'''
Ce = self.emb_enc(C, maskC, 1, 1) # l决定resnet的概率，blks=Block的个数 -> [16, 96, 400]
Qe = self.emb_enc(Q, maskQ, 1, 1) # [16, 96, 50]
X = self.cq_att(Ce, Qe, maskC, maskQ)

class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)#[96, 1]
        self.w4Q = nn.Parameter(w4Q)#[96, 1]
        self.w4mlu = nn.Parameter(w4mlu)#[1, 1, 96]
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2) #C->[16, 400, 96]
        Q = Q.transpose(1, 2) #Q->[16, 50, 96]
        batch_size_c = C.size()[0] # batch_size_c:16
        S = self.trilinear_for_attention(C, Q)

    def trilinear_for_attention(self, C, Q):#[16, 400, 96] [16, 50, 96]
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq]) # matmul([16, 400, 96], [96, 1])->[16, 400, 1]-expand->[16, 400, 50]
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1]) # [16, 50, 1] -> [16, 1, 50]-expand->[16, 400, 50]
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2)) #matmul([16, 400, 96], [16, 96, 50]) -> [16, 400, 50]
