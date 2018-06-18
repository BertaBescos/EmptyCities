-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'


opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   NSYNTH_DATA_ROOT = '',  -- path to non synthetic images (should have subfolders 'train', 'val', etc)
   batchSize = 1,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   input_mask_nc = 0,	   -- #  of input mask channels --bbescos
   maskD = 0,		   -- Penalize Dicriminator more on mask
   gamma = 2,
   niter = 200,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   gaussian_blur = 0,	   -- data augmentation
   gaussian_noise = 0,     -- data augmentation
   brightness = 0,	   -- data augmentation
   contrast = 0, 	   -- data augmentation
   saturation = 0,         -- data augmentation
   rotation = 0,	   -- data augmentation
   dropout = 0,		   -- data augmentation
   add_non_synthetic_data = 0,      -- train with real and synthetic data
   epoch_synth = 50,	   -- train with real and synthetic data
   pNonSynth = 0.1,	   -- train with real and synthetic data
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_plot = 'errL1',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   val_display_plot = 'val_errL1',
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, nsynth, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   val_freq = 5000,		-- see validation output every val_freq iteration
   save_epoch_freq = 50,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 10000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'unet',  -- selects model to use for netG
   n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
   lambda = 100,               -- weight on L1 term in objective
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

function pause ()
    print("Press any key to continue.")
    io.flush()
    io.read()
end

local input_nc = opt.input_nc
local output_nc = opt.output_nc
local input_mask_nc = opt.input_mask_nc --bbescos
local maskD = opt.maskD
-- translation direction
local idx_A = nil
local idx_B = nil
local idx_C = nil --bbescos

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc + 1, input_nc + output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

idx_C = {input_nc + output_nc + 1, input_nc + output_nc + input_mask_nc}

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
if input_mask_nc == 0 then
    data_loader = paths.dofile('data/data.lua')
else
    data_loader = paths.dofile('data/data2.lua') --bbescos
end
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

opt.phase = 'val'
local val_data = data_loader.new(opt.nThreads, opt)
print("Validation Dataset Size: ", val_data:size())

if opt.add_non_synthetic_data == 1 then
    opt.phase = 'train'
    nsynth_data_loader = paths.dofile('data/data_nsynth.lua') --bbescos
    nsynth_train_data = nsynth_data_loader.new(opt.nThreads, opt) --bbescos
    print("Non Synthetic Dataset Size: ", nsynth_train_data:size())
    opt.phase = 'val'
    nsynth_valid_data = nsynth_data_loader.new(opt.nThreads, opt) --bbescos
    print("Non Synthetic Validation Dataset Size: ", nsynth_valid_data:size())
end

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0
local synth_label = 1

function defineG(input_nc, output_nc, ngf)
    local netG = nil
    if     opt.which_model_netG == "encoder_decoder" then 
	netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet" then 
	netG = defineG_unet(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_128" then 
	netG = defineG_unet_128(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "unet_upsample" then
	netG = defineG_unet_upsampling(input_nc, output_nc, ngf)
    else 
	error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    if     opt.which_model_netD == "basic" then 
	netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then 
	netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end

-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
   print('define model netG...')
   netG = defineG(input_nc + input_mask_nc, output_nc, ngf)
   print('define model netD...')
   netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)

local criterion = nn.BCECriterion() --This is the Adversarial Criterion for the RGB image
local criterionAE = nn.AbsCriterion() --This is the L1 Loss for the RGB image
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local val_real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local nsynth_real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local nsynth_val_real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local val_real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local nsynth_real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local nsynth_val_real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local val_real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local nsynth_real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local nsynth_val_real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local val_fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local nsynth_fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local nsynth_val_fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1, errMASK = 0, 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   val_real_A = val_real_A:cuda();
   nsynth_real_A = nsynth_real_A:cuda();
   nsynth_val_real_A = nsynth_val_real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   val_real_B = val_real_B:cuda(); val_fake_B = val_fake_B:cuda();
   nsynth_real_B = nsynth_real_B:cuda(); nsynth_fake_B = nsynth_fake_B:cuda();
   nsynth_val_real_B = nsynth_val_real_B:cuda(); nsynth_val_fake_B = nsynth_val_fake_B:cuda();
   real_C = real_C:cuda();
   val_real_C = val_real_C:cuda();
   nsynth_real_C = nsynth_real_C:cuda();
   nsynth_val_real_C = nsynth_val_real_C:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda();
   print('done')
else
   print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    local val_data, val_data_path = val_data:getBatch()
    if synth_label == 0 then
	nsynth_data, nsynth_data_path = nsynth_train_data:getBatch()
	nsynth_val_data, nsynth_val_data_path = nsynth_valid_data:getBatch()
    end
    data_tm:stop()

    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    if input_mask_nc == 1 then
	real_C:copy(real_data[{ {}, idx_C, {}, {} }])
    end

    val_real_A:copy(val_data[{ {}, idx_A, {}, {} }])
    val_real_B:copy(val_data[{ {}, idx_B, {}, {} }])
    if input_mask_nc == 1 then
	val_real_C:copy(val_data[{ {}, idx_C, {}, {} }])
    end

    if synth_label == 0 then
       	nsynth_real_A:copy(nsynth_data[{ {}, idx_A, {}, {} }])
       	nsynth_real_B:copy(nsynth_data[{ {}, idx_B, {}, {} }])
       	if input_mask_nc == 1 then
	   nsynth_real_C:copy(nsynth_data[{ {}, idx_C, {}, {} }])
       	end
       	nsynth_val_real_A:copy(nsynth_val_data[{ {}, idx_A, {}, {} }])
	nsynth_val_real_B:copy(nsynth_val_data[{ {}, idx_B, {}, {} }])
	if input_mask_nc == 1 then
	   nsynth_val_real_C:copy(nsynth_val_data[{ {}, idx_C, {}, {} }])
	end
    end
    
    -- create fake

    if synth_label == 0 then
	real_A = nsynth_real_A:clone()
	real_B = nsynth_real_B:clone()
	real_C = nsynth_real_C:clone()
	real_C:zero()
	val_real_A = nsynth_val_real_A:clone()
	val_real_B = nsynth_val_real_B:clone()
	val_real_C = nsynth_val_real_C:clone()
	val_real_C:zero()
    end

    if opt.condition_GAN==1 then
	real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end   

    if input_mask_nc == 0 then
       fake_B = netG:forward(real_A)
    end
    if input_mask_nc == 1 then
       fake_B = netG:forward(torch.cat(real_A,real_C,2))
    end

    if opt.condition_GAN==1 then
	fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end

end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()

    -- Real
    local output = netD:forward(real_AB) -- Bajar el valor de la zona de la mascara  

    if synth_label == 1 then
        if maskD == 1 then
    	   mask_C = real_C:float()
    	   mask_C:resize(mask_C:size(3), mask_C:size(4))
    	   mask_C = image.scale(mask_C, output:size(3), output:size(4)) -- min=-1 y max=1
    	   mask_C = mask_C:add(1):mul(0.5)
    	   mask_C[mask_C:gt(0)] = 1
    	   mask_C:resize(1, 1, mask_C:size(1), mask_C:size(1))
    	   if opt.gpu > 0 then    
	      mask_C = mask_C:cuda()
    	   end
	   output[mask_C:eq(1)] = output[mask_C:eq(1)]:mul(-1):add(1):mul(-opt.gamma):add(1)
	   output[output:lt(0)] = 0
        end

        label = torch.FloatTensor(output:size()):fill(real_label)
        if opt.gpu>0 then 
    	   label = label:cuda()
        end
        errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(real_AB, df_do)
    else
        label = torch.FloatTensor(output:size()):fill(real_label)
        if opt.gpu>0 then 
    	   label = label:cuda()
        end
        errD_real = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        netD:backward(real_AB, df_do)
    end
    
    -- Fake
    local output = netD:forward(fake_AB) -- Subir el valor de la zona de la mascara
    if synth_label == 1 then
        if maskD == 1 then
           aux_C = real_C:float()
    	   aux_C:resize(aux_C:size(3), aux_C:size(4))
    	   aux_C = image.scale(aux_C, output:size(3), output:size(4)) -- min=-1 y max=1
    	   mask_C = torch.Tensor(aux_C:size(1),aux_C:size(2)):zero():add(1)
    	   mask_C[aux_C:gt(-1)] = opt.gamma -- Parametros que tocar!!
    	   mask_C:resize(1, 1, mask_C:size(1), mask_C:size(1))
    	   if opt.gpu > 0 then    
	      mask_C = mask_C:cuda()
    	   end
    	   output = torch.cmul(output,mask_C)
    	   output[output:gt(1)] = 1
        end
        label:fill(fake_label)
        errD_fake = criterion:forward(output, label)
        df_do = criterion:backward(output, label)
        netD:backward(fake_AB, df_do)
    else
	label:fill(fake_label)
        errD_fake = criterion:forward(output, label)
        df_do = criterion:backward(output, label)
        netD:backward(fake_AB, df_do)
    end

    errD = (errD_real + errD_fake)/2

    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    if synth_label == 1 then
	if opt.use_GAN==1 then
	   local output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
	   if maskD == 1 then
	      mask_C = real_C:float()
    	      mask_C:resize(mask_C:size(3), mask_C:size(4))
    	      mask_C = image.scale(mask_C, output:size(3), output:size(4)) -- min=-1 y max=1
    	      mask_C = mask_C:add(1):mul(0.5)
    	      mask_C[mask_C:gt(0)] = 1
    	      mask_C:resize(1, 1, mask_C:size(1), mask_C:size(1))
    	      if opt.gpu > 0 then    
	      	mask_C = mask_C:cuda()
    	      end
	      output[mask_C:eq(1)] = output[mask_C:eq(1)]:mul(-1):add(1):mul(-opt.gamma):add(1)
	      output[output:lt(0)] = 0
	   end
	   local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
	   if opt.gpu>0 then 
	     label = label:cuda();
	   end
	   errG = criterion:forward(output, label)
	   local df_do = criterion:backward(output, label)
	   df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
	else
	   errG = 0
	end
    else
	if opt.use_GAN==1 then
	   local output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
	   --output[aux_C:gt(-1)] = real_label
           local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
	   if opt.gpu>0 then 
	     label = label:cuda();
	   end
	   errG = criterion:forward(output, label)
	   local df_do = criterion:backward(output, label)
	   df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
	else
	   errG = 0
	end
    end
    
    -- unary loss
    local df_do_AE = torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_AE = df_do_AE:cuda();
    end

    if synth_label == 1 then
       if opt.use_L1==1 then
          errL1 = criterionAE:forward(fake_B, real_B)
          df_do_AE = criterionAE:backward(fake_B, real_B)
       else
          errL1 = 0
       end
    else   
       if opt.use_L1==1 then
          --[[mask_C = real_C:float()
    	  if opt.gpu > 0 then    
	     mask_C = mask_C:cuda()
    	  end
          fake_B[mask_C:eq(1)] = real_B[mask_C:eq(1)]	
          --]]
	  errL1 = criterionAE:forward(fake_B, real_B)
          df_do_AE = criterionAE:backward(fake_B, real_B)
       else
          errL1 = 0
       end
    end
   
    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))   

    return errG, gradParametersG
end

-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({"errG", "errD", "errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- parse diplay_plot string into table
opt.val_display_plot = string.split(string.gsub(opt.val_display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.val_display_plot) do
    if not util.containsValue({"val_errG", "val_errD", "val_errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot), unpack(opt.val_display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch

	if opt.add_non_synthetic_data == 1 and epoch > opt.epoch_synth then
	   if torch.uniform() > opt.pNonSynth then
	      synth_label = 1
	   else
	      synth_label = 0
	   end
	end

        createRealFake()

	--winqt0 = image.display{image=real_A, win=winqt0}
	--winqt1 = image.display{image=real_B, win=winqt1}
	--winqt2 = image.display{image=fake_B, win=winqt2}
	--pause()

        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end

        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        optim.adam(fGx, parametersG, optimStateG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then

            createRealFake()

            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
		if input_nc == 3 and input_mask_nc == 0 then
			disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
			disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
			disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target'})
		end
		if input_nc == 1 and input_mask_nc == 0 then
			local img_input = util.scaleBatch(real_A:float(),100,100)
			img_input = img_input:add(1):div(2)
			disp.image(img_input, {win=opt.display_id, title=opt.name .. ' input'})
			local img_output = util.scaleBatch(fake_B:float(),100,100)
			img_output = img_output:add(1):div(2)
			disp.image(img_output, {win=opt.display_id+1, title=opt.name .. ' output'})
			local img_target = util.scaleBatch(real_B:float(),100,100)
			img_target = img_target:add(1):div(2)
			disp.image(img_target, {win=opt.display_id+2, title=opt.name .. ' target'})
		end
		if input_mask_nc == 1 and input_nc == 3 then
			disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
			local mask_input = util.scaleBatch(real_C:float(),100,100)
			mask_input = mask_input:add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' input_MASK'})
			disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' output'})
			disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+3, title=opt.name .. ' target'})
		end
		if input_mask_nc == 1 and input_nc == 1 then
			local img_input = util.scaleBatch(real_A:float(),100,100)
			img_input = img_input:add(1):div(2)
			disp.image(img_input, {win=opt.display_id, title=opt.name .. ' input'})
			local mask_input = util.scaleBatch(real_C:float(),100,100)
			mask_input = mask_input:add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' input_MASK'})
			local img_output = util.scaleBatch(fake_B:float(),100,100)
			img_output = img_output:add(1):div(2)
			disp.image(img_output, {win=opt.display_id+2, title=opt.name .. ' output'})
			local img_target = util.scaleBatch(real_B:float(),100,100)
			img_target = img_target:add(1):div(2)
			disp.image(img_target, {win=opt.display_id+3, title=opt.name .. ' target'})
		end
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then 
				if input_nc == 3 then
					image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
				end
				if input_nc == 1 then
					local aux_A = real_A[i2]:float()
					aux_A = aux_A:add(1):div(2)
					local aux_B = fake_B[i2]:float()
					aux_B = aux_B:add(1):div(2)
					image_out = torch.cat(aux_A,aux_B,3)
				end
                        else
				if input_nc == 3 then
					image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) 
				end
				if input_nc == 1 then
					local aux_A = real_A[i2]:float()
					aux_A = aux_A:add(1):div(2)
					local aux_B = fake_B[i2]:float()
					aux_B = aux_B:add(1):div(2)
					image_out = torch.cat(image_out, torch.cat(aux_A,aux_B,3), 2) 
				end
			end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
	if (counter % opt.val_freq == 0 or counter == 1) and opt.display then
	    createRealFake()
	    val_fake_B = netG:forward(torch.cat(val_real_A,val_real_C,2))
	    val_errL1 = criterionAE:forward(val_fake_B, val_real_B)
	    if input_nc == 3 and input_mask_nc == 0 then
		disp.image(util.deprocess_batch(util.scaleBatch(val_real_A:float(),100,100)), {win=opt.display_id+3, title=opt.name .. ' val_input'})
		disp.image(util.deprocess_batch(util.scaleBatch(val_fake_B:float(),100,100)), {win=opt.display_id+4, title=opt.name .. ' val_output'})
		disp.image(util.deprocess_batch(util.scaleBatch(val_real_B:float(),100,100)), {win=opt.display_id+5, title=opt.name .. ' val_target'})
	    end
	    if input_nc == 1 and input_mask_nc == 0 then
		local img_input = util.scaleBatch(val_real_A:float(),100,100)
		img_input = img_input:add(1):div(2)
		disp.image(img_input, {win=opt.display_id+3, title=opt.name .. ' val_input'})
		local img_output = util.scaleBatch(val_fake_B:float(),100,100)
		img_output = img_output:add(1):div(2)
		disp.image(img_output, {win=opt.display_id+4, title=opt.name .. ' val_output'})
		local img_target = util.scaleBatch(val_real_B:float(),100,100)
		img_target = img_target:add(1):div(2)
		disp.image(img_target, {win=opt.display_id+5, title=opt.name .. ' val_target'})
	    end
	    if input_mask_nc == 1 and input_nc == 3 then
		disp.image(util.deprocess_batch(util.scaleBatch(val_real_A:float(),100,100)), {win=opt.display_id+4, title=opt.name .. ' val_input'})
		local mask_input = util.scaleBatch(val_real_C:float(),100,100)
		mask_input = mask_input:add(1):div(2)
		disp.image(mask_input, {win=opt.display_id+5, title=opt.name .. ' val_input_MASK'})
		disp.image(util.deprocess_batch(util.scaleBatch(val_fake_B:float(),100,100)), {win=opt.display_id+6, title=opt.name .. ' val_output'})
		disp.image(util.deprocess_batch(util.scaleBatch(val_real_B:float(),100,100)), {win=opt.display_id+7, title=opt.name .. ' val_target'})
	    end
	    if input_mask_nc == 1 and input_nc == 1 then
		local img_input = util.scaleBatch(val_real_A:float(),100,100)
		img_input = img_input:add(1):div(2)
		disp.image(img_input, {win=opt.display_id+4, title=opt.name .. ' val_input'})
		local mask_input = util.scaleBatch(val_real_C:float(),100,100)
		mask_input = mask_input:add(1):div(2)
		disp.image(mask_input, {win=opt.display_id+5, title=opt.name .. ' val_input_MASK'})
		local img_output = util.scaleBatch(val_fake_B:float(),100,100)
		img_output = img_output:add(1):div(2)
		disp.image(img_output, {win=opt.display_id+6, title=opt.name .. ' val_output'})
		local img_target = util.scaleBatch(val_real_B:float(),100,100)
		img_target = img_target:add(1):div(2)
		disp.image(img_target, {win=opt.display_id+7, title=opt.name .. ' val_target'})
	    end


	end

        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
	    local val_errG = nil
	    local val_errD = nil
	    local val_loss = {val_errG=val_errG and val_errG or -1, val_errD=val_errD and val_errD or -1, val_errL1=val_errL1 and val_errL1 or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrL1: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG, errD, errL1))
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              	if loss[v] ~= nil then
                   plot_vals[#plot_vals + 1] = loss[v]
              	end
	    end
	    for k, v in ipairs(opt.val_display_plot) do
              	if val_loss[v] ~= nil then
                   plot_vals[#plot_vals + 1] = val_loss[v]
              	end
	    end
            -- update display plot
            if opt.display then
              table.insert(plot_data, plot_vals)
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end

        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
        end
        
    end
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil

    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
end
