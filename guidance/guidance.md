# CPEN455 Assignment Guidance: PixelCNN++
This guide will walk you through the process of understanding, implementing, training, and evaluating PixelCNN++ models. **Note that this guidance is intended as a reference - you should refer to the codebase and README.md requirements to implement your own solution.**

## Step 1: Environment Setup and Initial Run

### 1.1 Setting Up Your Environment
- Install all required dependencies using the provided `requirements.txt` or follow the setup instructions in README.md.
- It's recommended to have a GPU environment ready, as training PixelCNN++ models can be computationally intensive. If you don't have a GPU, CPU can also be used but training will take significantly longer.
- If using Weights & Biases (WandB), create an account and obtain your API key.

### 1.2 Running the Initial Training Script
- Execute this training script if you **want to train the model with WandB**:
  ```bash
  python pcnn_train.py --batch_size 16 --sample_batch_size 16 --sampling_interval 25 --save_interval 25 --dataset cpen455 --nr_resnet 1 --lr_decay 0.999995 --max_epochs 100 --en_wandb True
  ```
- Execute this training script if you **don't want to use WandB**:
  ```bash
  python pcnn_train.py --batch_size 16 --sample_batch_size 16 --sampling_interval 25 --save_interval 25 --dataset cpen455 --nr_resnet 1 --lr_decay 0.999995 --max_epochs 100
  ```
- If using WandB, you should see training metrics appearing on your dashboard.
- Pay special attention to the Bits Per Dimension (BPD) metric during training, as it is a standard metric for evaluating likelihood-based generative models.
<details>
    <summary>Understanding BPD</summary>

Here's an explanation of BPD (Bits Per Dimension):

**Bits Per Dimension (BPD)** is a fundamental metric for evaluating autoregressive generative models like PixelCNN++. From the code and instructions:

1. **Calculation** in `pcnn_train.py`:
```python
# The BPD calculation appears in the training loop:
deno = args.batch_size * np.prod(args.obs) * np.log(2.)
loss_tracker.update(loss.item()/deno)
```
This implements the formula:  
`BPD = -ln(likelihood) / (dimensions * ln(2))`

2. **Key Properties**:
- Lower BPD = Better performance (model needs fewer bits to encode each pixel dimension)
- Directly relates to model's negative log-likelihood
- Normalized by image dimensions, making it comparable across different datasets

3. **Comparison with Other Metrics**:
- Unlike FID or Inception Score which measure sample quality, BPD directly measures likelihood assignment

4. **Monitoring** in `pcnn_train.py`:
```python
# BPD is logged to WandB during training:
wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
```
</details>

<details>
    <summary>Generating Samples</summary>

The sample function has already been implemented in the codebase. The exact sampling implementation can be found in `lines 177-188 in utils.py`:

```python
def sample(model, sample_batch_size, obs, sample_op):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(next(model.parameters()).device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = data
                out   = model(data_v, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data
```

This is the actual sample function used in the training loop. The key implementation details are:

1. The function initializes a zero tensor with shape (batch_size, channels, height, width)
2. It autoregressively generates pixels row by row, column by column
3. Uses the `sample_op` (defined as `sample_from_discretized_mix_logistic`) to get actual pixel values from the model's output distribution

The sample function is called in `lines 221-225 in pcnn_train.py`:

```python
        if epoch % args.sampling_interval == 0:
            print('......sampling......')
            sample_t = sample(model, args.sample_batch_size, args.obs, sample_op)
            sample_t = rescaling_inv(sample_t)
            save_images(sample_t, args.sample_dir)
```

The sampling process uses the model in eval mode and generates pixels sequentially using the discretized mixture of logistics output distribution defined in `lines 112-152 in utils.py`.
</details>

## Step 2: Understanding the PixelCNN++ Codebase

### 2.1 Model Architecture Overview
PixelCNN++ utilizes a U-Net-like architecture with specialized components for autoregressive generation:

#### U-Net Architecture Fundamentals
The codebase shows a U-Net-like structure through its up/down sampling layers and skip connections, in `lines 5-50 in model.py`:

```python
class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul
```

Key components:
1. **Contracting Path (Encoder)**
   - Uses `PixelCNNLayer_down` with downsampling via `downsize_u_stream`
   - Progressive reduction of spatial resolution (lines 75-85 in model.py)

2. **Expanding Path (Decoder)**
   - Uses `PixelCNNLayer_up` with upsampling via `upsize_u_stream`
   - Restores spatial resolution (lines 81-85 in model.py)

3. **Skip Connections**
   - Implemented through `u_list` and `ul_list` buffers
   - Connects corresponding encoder-decoder layers (lines 47-48 in model.py)

### Fusion Strategies in the Architecture
**The most common fusion implementation is to add two tensors and pass them to the remaining network.** To be more specific, in our project, the most common fusion implementation is to add the class embedding(the embedding of the class label, which is commonly the output of a embedding layer) to the input of some layers and pass them through the remaining network. Generally, to modify the unconditional model to be conditional, here are the steps:
  1. Add an embedding layer for your condition (e.g., class labels):
     ```python
     self.embedding = nn.Embedding(num_classes, embedding_dim)
     ```
  
  2. Incorporate this embedding throughout the network. Common methods include:
     - Adding the embedding to each layer's input
     - Using FiLM layers (Feature-wise Linear Modulation)
     - Concatenating the embedding to intermediate features
  
  3. A simple implementation might look like:
     ```python
     class_embedding = self.embedding(class_labels)
     # Reshape embedding for broadcasting
     class_embedding = class_embedding.view(batch_size, embedding_dim, 1, 1)
     # Add to feature maps
     x = x + class_embedding

Here are some suggestions for the fusion strategies in the architecture:

**1. Early Fusion**
- Input conditioning can be done at network entry in `line 100 in model.py`, before the input passed the whole network.

**2. Middle Fusion**
- Insert the fusion condition in the U-net in `lines 113-140 in model.py`:

```python
        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

```

**3. Fuse in Gated Residual Blocks**
- Gated residual blocks enable deep feature integration in `lines 118-141 in layers.py`:

```python
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu

        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3
```

**4.Late Fusion**

Add the fusion condition to the output of the U-net.


## Step 3: Training and Evaluating Conditional PixelCNN++

### 3.1 Training Process
- Train the model for a sufficient number of epochs (depends on the model size and the dataset, you should monitor the training and validation BPD, and stop the training when the validation BPD is no longer decreasing).
- Save checkpoints periodically.

### 3.2 Hyperparameter Tuning
- Experiment with different learning rates, batch sizes, and model sizes.(in our code base, the most efficient hyperparameter to adjust the model size is the number of resnet layers,the number of filters and the number of logistic mix, you can try different combinations and find the one fits your computation resource)
- Record the effect of these changes on training speed and final performance.

### 3.3 Evaluating with BPD
- Calculate BPD on a held-out test set to evaluate generalization.
- Compare your model's BPD to published benchmarks for the dataset you're using.

### 3.4 Sample Quality Assessment
- Generate multiple sample batches to assess variety and quality.
- Consider using FID (Fréchet Inception Distance) or Inception Score if available to quantitatively assess sample quality.

## Step 4: Transforming Conditional PixelCNN++ into a Classifier

### 4.1 Classification Approach
- Once you have a trained conditional PixelCNN++, you can use Bayes' rule to perform classification:
  ```
  P(class|image) ∝ P(image|class) × P(class)
  ```
- For each class, compute the likelihood of the image under your conditional model.
- The predicted class is the one that maximizes this likelihood (assuming a uniform prior over classes).

### 4.2 Implementation
- Write a function that:
  1. Takes an image as input
  2. For each possible class label:
     - Conditions the model on that label
     - Computes the likelihood (or negative BPD) of the image
  3. Returns the class with the highest likelihood

### 4.3 the code of the log-likelihood of PixelCNN++
The log-likelihood calculation is implemented in the `discretized_mix_logistic_loss` function from `lines 36-101 in utils.py`:

```python
def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
   
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    
    return -torch.sum(log_sum_exp(log_probs))
```


This function calculates the negative log-likelihood. To get the actual log-likelihood, you would use:

```python
# During evaluation
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    neg_log_likelihood = discretized_mix_logistic_loss(inputs, outputs)
    log_likelihood = -neg_log_likelihood
```

The key implementation details are:
1. The loss function handles the mixture of logistics distribution parameters coming from the model (`outputs`)
2. It computes the log-probability for each sub-pixel value in `inputs`
3. The final loss is the negative sum of log-probabilities

This is already used in the training loop in `lines 17-39 in pcnn_train.py`:


```python
def train_or_test(model, data_loader, optimizer, loss_op, device, args, epoch, mode = 'training'):
    if mode == 'training':
        model.train()
    else:
        model.eval()
        
    deno =  args.batch_size * np.prod(args.obs) * np.log(2.)        
    loss_tracker = mean_tracker()
    
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, _ = item
        model_input = model_input.to(device)
        model_output = model(model_input)
        loss = loss_op(model_input, model_output)
        loss_tracker.update(loss.item()/deno)
        if mode == 'training':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    if args.en_wandb:
        wandb.log({mode + "-Average-BPD" : loss_tracker.get_mean()})
        wandb.log({mode + "-epoch": epoch})
```


But note the training code divides by `deno` to convert to BPD (Bits Per Dimension). For raw log-likelihood, you would skip that division and just use the direct output from `discretized_mix_logistic_loss` multiplied by -1.


## Step 5: Analysis and Improvement

### 5.1 Analyzing Results
- Use the BPD to analyze the performance of your model.
- Use the FID score to assess the quality of the generated samples.
- Use the accuracy of your model to analyze the performance.

### 5.2 Model Structure Improvements
- Experiment with deeper or wider networks
- Try different residual block configurations
### 5.3 Conditioning Mechanism Improvements
- Compare different ways of incorporating conditions:
  - Addition vs. concatenation vs. FiLM conditioning
  - Conditioning at different depths of the network
  - Using multiple conditioning points

### 5.4 Training Improvements
- Try different optimizers (Adam, AdamW, SGD with momentum)
- Experiment with learning rate schedules
- Data augmentation strategies
- Longer training runs
