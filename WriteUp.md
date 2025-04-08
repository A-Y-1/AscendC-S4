### Eye

复制nrows个dataBlock(32byte)的1，dataCopyPad nrows个1至目标位置。优化空间较小。另外需要注意nrows是否大于ncolumns，但线上测试没有对该种case的测试。

### Heaviside

共有三种情况：

* values为标量
* values多维度，但与input不一致
* values多维度且与input维度一致

其中第一种和第三种情况都可以简单对input进行划分，按照32Byte对齐划分给每个核，核内再根据UB大小多次迭代处理。第二种情况简单按照了batch数量进行划分给每个核，优化空间仍然比较大，可以每次搬入和处理多个batch，从而避免batch数量大影响性能，仓库代码未做该优化。

### MatMul

