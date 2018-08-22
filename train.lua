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
require 'cudnn'
--require 'imgraph'

opt = {
	DATA_ROOT = '',			-- path to images (should have subfolders 'train', 'val', etc)
	NSYNTH_DATA_ROOT = '',	-- path to non synthetic images (should have subfolders 'train', 'val', etc)
	batchSize = 1,          -- # images in batch
	loadSize = 286,         -- scale images to this size
	fineSize = 256,         -- then crop to this size
	mask = 1,				-- set to 1 if CARLA images have mask (always on training)
	target = 1,				-- set to 1 if CARLA images have target (always on training)
	ngf = 64,               -- #  of gen filters in first conv layer
	ndf = 64,               -- #  of discrim filters in first conv layer
	input_nc = 3,           -- #  of input image channels
	output_nc = 3,          -- #  of output image channels
	input_mask_nc = 1,	   -- #  of input mask channels --bbescos
	input_gan_nc = 1,			-- #  of input image channels to the pix2pix architecture
	output_gan_nc = 1,	   -- #  of output image channels from the pix2pix architecture
	mGAN = 1,		   		-- Penalize Dicriminator more on mask
	gamma = 2,					-- Penalize Dicriminator two more times on mask
	niter = 200,            -- #  of iter at starting learning rate
	lr = 0.0002,            -- initial learning rate for adam
	beta1 = 0.5,            -- momentum term of adam
	ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
	data_aug = 0,	   		-- data augmentation
	epoch_synth = 0,	   	-- train with real and synthetic data from this epoch on
	pNonSynth = 0.5,	   	-- train with real and synthetic data with this proportion
	display = 1,            -- display samples while training. 0 = false
	display_id = 10,        -- display window id.
	display_plot = 'errL1', 	-- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
	val_display_plot = 'val_errL1',
	gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
	name = 'mGAN',           -- name of the experiment, should generally be passed on the command line
	phase = 'train',             	-- train, val, test, nsynth, etc
	preprocess = 'regular',      	-- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
	nThreads = 2,                	-- # threads for loading data
	val_freq = 5000,		-- see validation output every val_freq iteration
	save_epoch_freq = 50,        	-- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
	save_latest_freq = 5000,     	-- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
	print_freq = 50,             	-- print the debug information every print_freq iterations
	display_freq = 100,          	-- display the current results every display_freq iterations
	save_display_freq = 10000,    -- save the current display of results every save_display_freq_iterations
	continue_train=0,            	-- if continue training, load the latest model: 1: true, 0: false
	epoch_ini = 1,		-- if continue training, at what epoch we start
	serial_batches = 0,          	-- if 1, takes images in order to make batches, otherwise takes them randomly
	serial_batch_iter = 1,       	-- iter into serial image list
	checkpoints_dir = './checkpoints', 	-- models are saved here
	cudnn = 1,                         	-- set to 0 to not use cudnn
	condition_GAN = 1,                 	-- set to 0 to use unconditional discriminator
	use_GAN = 1,                       	-- set to 0 to turn off GAN term
	use_L1 = 1,                   		-- set to 0 to turn off L1 term
	which_model_netD = 'basic', 	-- selects model to use for netD
	which_model_netG = 'unet',  	-- selects model to use for netG
	n_layers_D = 0,             	-- only used if which_model_netD=='n_layers'
	lambda = 100,               	-- weight on L1 term in objective
	lambdaSS = 1,		       		-- weight on SS term in objective
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- useful function for debugging
function pause ()
	 print("Press any key to continue.")
	 io.flush()
	 io.read()
end

local input_nc = opt.input_nc
local output_nc = opt.output_nc
local input_mask_nc = opt.input_mask_nc --bbescos
local mGAN = opt.mGAN
local input_gan_nc = opt.input_gan_nc
local output_gan_nc = opt.output_gan_nc

-- translation direction
local idx_A = nil
local idx_B = nil
local idx_C = nil

idx_A = {1, input_nc}
idx_B = {input_nc + 1, input_nc + output_nc}
idx_C = {input_nc + output_nc + 1, input_nc + output_nc + input_mask_nc}

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader for CARLA images (train and val)
local synth_data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local synth_data = synth_data_loader.new(opt.nThreads, opt)
print("CARLA Dataset Size: ", synth_data:size())
opt.phase = 'val'
local val_synth_data = synth_data_loader.new(opt.nThreads, opt)
print("Validation CARLA Dataset Size: ", val_synth_data:size())

-- create data loader for real images (train and val)
if opt.NSYNTH_DATA_ROOT ~= '' then
	 opt.phase = 'train'
	 nsynth_data_loader = paths.dofile('data/data_nsynth.lua') --bbescos
	 nsynth_data = nsynth_data_loader.new(opt.nThreads, opt) --bbescos
	 print("Non Synthetic Dataset Size: ", nsynth_data:size())
	 opt.phase = 'val'
	 val_nsynth_data = nsynth_data_loader.new(opt.nThreads, opt) --bbescos
	 print("Non Synthetic Validation Dataset Size: ", val_nsynth_data:size())
end

opt.phase = train

----------------------------------------------------------------------------

-- function for initializing model weights
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

-- function to load generator G
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

-- function to load discriminetor D
function defineD(input_nc, output_nc, ndf)
	local netD = nil
	if opt.condition_GAN==1 then
		input_nc_tmp = input_nc
	else
		input_nc_tmp = 0 -- only penalizes structure in output channels
	end
	if opt.which_model_netD == "basic" then 
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
	if opt.NSYNTH_DATA_ROOT ~= '' then
		if opt.epoch_ini > opt.epoch_synth then
			print('loading previously trained netSS...')
			netSS = torch.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_SS.net'), opt)
			netSS:training()
		else
			print('define model netSS...')
			local ss_path = "./checkpoints/SemSeg/erfnet.net"
			netSS = torch.load(ss_path)
			netSS:training()
		end
	end
else
	print('define model netG...')
	netG = defineG(input_gan_nc + input_mask_nc, output_gan_nc, ngf)
	print('define model netD...')
	netD = defineD(input_gan_nc, output_gan_nc, ndf)
	if opt.NSYNTH_DATA_ROOT ~= '' then
		print('define model netSS...')
		local ss_path = "./checkpoints/SemSeg/erfnet.net"
		netSS = torch.load(ss_path)
		netSS:training()
	end
end

-- define netDynSS model 
if opt.NSYNTH_DATA_ROOT ~= '' then
	print('define model netDynSS...')
	netDynSS = nn.Sequential()
	local convDyn = nn.SpatialFullConvolution(20,1,1,1,1,1)
	local w, dw = convDyn:parameters()
	w[1][{{1,12}}] = -8/20 -- Static
	w[1][{{13,20}}] = 12/20 -- Dynamic
	w[2]:fill(0)
	netDynSS:add(nn.SoftMax())
	netDynSS:add(convDyn):add(nn.MulConstant(100))
	netDynSS:add(nn.Tanh())
end

-- define criteria
local criterion = nn.BCECriterion() --This is the Adversarial Criterion
local criterionAE = nn.AbsCriterion() --This is the L1 Loss
if opt.NSYNTH_DATA_ROOT ~= '' then
	local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence','Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'}
	local classWeights = torch.Tensor(#classes)
	classWeights[1] = 0.0				-- unkown
	classWeights[2] = 2.8149201869965	-- road
	classWeights[3] = 6.9850029945374	-- sidewalk
	classWeights[4] = 3.7890393733978	-- building
	classWeights[5] = 9.9428062438965	-- wall
	classWeights[6] = 9.7702074050903	-- fence
	classWeights[7] = 9.5110931396484	-- pole
	classWeights[8] = 10.311357498169	-- traffic light
	classWeights[9] = 10.026463508606	-- traffic sign
	classWeights[10] = 4.6323022842407	-- vegetation
	classWeights[11] = 9.5608062744141	-- terrain
	classWeights[12] = 7.8698215484619	-- sky
	classWeights[13] = 9.5168733596802	-- person
	classWeights[14] = 10.373730659485	-- rider
	classWeights[15] = 6.6616044044495	-- car
	classWeights[16] = 10.260489463806	-- truck
	classWeights[17] = 10.287888526917	-- bus
	classWeights[18] = 10.289801597595	-- train
	classWeights[19] = 10.405355453491	-- motorcycle
	classWeights[20] = 10.138095855713	-- bicycle
	criterionSS = cudnn.SpatialCrossEntropyCriterion(classWeights)
end

---------------------------------------------------------------------------

optimStateG = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
optimStateD = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
if opt.NSYTNH_DATA_ROOT ~= '' then
	optimStateSS = {
		learningRate = opt.lr,
		beta1 = opt.beta1,
	}
end
----------------------------------------------------------------------------

local realRGB_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local val_realRGB_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local realRGB_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local val_realRGB_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local val_real_C = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize) --bbescos
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local val_fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1, errSS = 0, 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

----------------------------------------------------------------------------

if opt.gpu > 0 then
	print('transferring to gpu...')
	require 'cunn'
	cutorch.setDevice(opt.gpu)
	realRGB_A = realRGB_A:cuda()
	val_realRGB_A = val_realRGB_A:cuda()
	realRGB_B = realRGB_B:cuda(); fake_B = fake_B:cuda()
	val_realRGB_B = val_realRGB_B:cuda(); val_fake_B = val_fake_B:cuda()
	real_C = real_C:cuda()
	val_real_C = val_real_C:cuda()
	real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda()
	if opt.cudnn==1 then
		netG = util.cudnn(netG); netD = util.cudnn(netD)
	end
	netD:cuda(); netG:cuda(); 
	criterion:cuda(); criterionAE:cuda(); 
	if opt.NSYNTH_DATA_ROOT ~= '' then
		netDynSS:cuda()
		criterionSS:cuda()
	end
	print('done')
else
	print('running model on CPU')
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
if opt.NSYNTH_DATA_ROOT ~= '' then
	parametersSS, gradParametersSS = netSS:getParameters()
end
if opt.display then disp = require 'display' end

----------------------------------------------------------------------------

function createRealFake()
	 -- load real
	data_tm:reset(); data_tm:resume()
	if synth_label == 1 then -- CARLA images
		real_data, data_path = synth_data:getBatch()
	else -- CITYSCAPES images
		real_data, data_path = nsynth_data:getBatch()
	end
	data_tm:stop()

	realRGB_A:copy(real_data[{ {}, idx_A, {}, {} }])
	realRGB_B:copy(real_data[{ {}, idx_B, {}, {} }])
	real_C:copy(real_data[{ {}, idx_C, {}, {} }]) --if CARLA it is dynamic

	-- crete mask
	if synth_label == 0 then
		realBGR_A = realRGB_A:clone():add(1):mul(0.5)
		realBGR_A[1][1] = realRGB_A[1][3]:add(1):mul(0.5)
		realBGR_A[1][3] = realRGB_A[1][1]:add(1):mul(0.5)
		erfnet_C = netSS:forward(realBGR_A) --20 channels
		fake_C = netDynSS:forward(erfnet_C)
		_,winner = erfnet_C:squeeze():max(1)
		winner:resize(1,winner:size(1),winner:size(2),winner:size(3))
	else
		fake_C = real_C:clone()
	end 
	
	-- convert A and B to gray scale
	realGray_A = image.rgb2y(realRGB_A[1]:float())
	realGray_A = realGray_A:cuda()
	realGray_A = realGray_A:resize(1,realGray_A:size(1),realGray_A:size(2),realGray_A:size(3))
	realGray_B = image.rgb2y(realRGB_B[1]:float())
	realGray_B = realGray_B:cuda()
	realGray_B = realGray_B:resize(1,realGray_B:size(1),realGray_B:size(2),realGray_B:size(3))

	-- create fake
	if opt.condition_GAN==1 then
		real_AB = torch.cat(realGray_A,realGray_B,2)
	else
		real_AB = realGray_B -- unconditional GAN, only penalizes structure in B
	end   

	fake_B = netG:forward(torch.cat(realGray_A,fake_C,2))
	
	if opt.condition_GAN==1 then
		fake_AB = torch.cat(realGray_A,fake_B,2)
	else
		fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
	end
end

function val_createRealFake()
	 -- load real
	data_tm:reset(); data_tm:resume()
	if synth_label == 1 then -- CARLA images
		val_data, val_data_path = val_synth_data:getBatch()
	else -- CITYSCAPES images
		val_data, val_data_path = val_nsynth_data:getBatch()
	end
	data_tm:stop()

	val_realRGB_A:copy(val_data[{ {}, idx_A, {}, {} }])
	val_realRGB_B:copy(val_data[{ {}, idx_B, {}, {} }])
	val_real_C:copy(val_data[{ {}, idx_C, {}, {} }]) --if CARLA it is dynamic
	
	-- crete mask
	if synth_label == 0 then
		val_realBGR_A = val_realRGB_A:clone():add(1):mul(0.5)
		val_realBGR_A[1][1] = val_realRGB_A[1][3]:add(1):mul(0.5)
		val_realBGR_A[1][3] = val_realRGB_A[1][1]:add(1):mul(0.5)
		val_erfnet_C = netSS:forward(val_realBGR_A) --20 channels
		val_fake_C = netDynSS:forward(val_erfnet_C)
		_,val_winner = val_erfnet_C:squeeze():max(1)
		val_winner:resize(1,val_winner:size(1),val_winner:size(2),val_winner:size(3))
	else
		val_fake_C = val_real_C:clone()
	end 
	
	-- convert A and B to gray scale
	val_realGray_A = image.rgb2y(val_realRGB_A[1]:float())
	val_realGray_A = val_realGray_A:cuda()
	val_realGray_A = val_realGray_A:resize(1,val_realGray_A:size(1),val_realGray_A:size(2),val_realGray_A:size(3))
	val_realGray_B = image.rgb2y(val_realRGB_B[1]:float())
	val_realGray_B = val_realGray_B:cuda()
	val_realGray_B = val_realGray_B:resize(1,val_realGray_B:size(1),val_realGray_B:size(2),val_realGray_B:size(3))

	-- create fake
	if opt.condition_GAN==1 then
		val_real_AB = torch.cat(val_realGray_A,val_realGray_B,2)
	else
		val_real_AB = val_realGray_B -- unconditional GAN, only penalizes structure in B
	end   

	val_fake_B = netG:forward(torch.cat(val_realGray_A,val_fake_C,2))
	
	if opt.condition_GAN==1 then
		val_fake_AB = torch.cat(val_realGray_A,val_fake_B,2)
	else
		val_fake_AB = val_fake_B -- unconditional GAN, only penalizes structure in B
	end
end


----------------------------------------------------------------------------

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
	netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
	 
	gradParametersD:zero()

	-- Real
	local output = netD:forward(real_AB) -- 1x1x30x30  
	label = torch.FloatTensor(output:size()):fill(real_label)
	if opt.gpu>0 then 
		label = label:cuda()
	end

	if synth_label == 1 then
		if mGAN == 1 then
			m = real_C:float():resize(real_C:size(3), real_C:size(4))
			m = image.scale(m, output:size(3), output:size(4)):add(1):mul(0.5)
			m[m:gt(0)] = 1
			m:resize(1, 1, m:size(1), m:size(2))
			local layer1 = nn.CMul(output:size())
			local term_layer1 = m:clone():mul(opt.gamma-1):add(1)
			layer1.weight = term_layer1
			local layer2 = nn.CAdd(output:size())
			local term_layer2 = m:clone():mul(1-opt.gamma)
			layer2.bias = term_layer2
			local layer3 = nn.ReLU()
			if opt.gpu > 0 then
				layer1 = layer1:cuda()
				layer2 = layer2:cuda()
				layer3 = layer3:cuda()
			end
			local mGAN_D = nn.Sequential()
			mGAN_D:add(layer1):add(layer2):add(layer3)
			local output_mGAN = mGAN_D:forward(output)
			errD_real = criterion:forward(output_mGAN, label)
			local df_do = criterion:backward(output_mGAN, label) -- 1x1x30x30  
			local df_dg = mGAN_D:updateGradInput(output, df_do)
			netD:backward(real_AB, df_dg)
		else
			errD_real = criterion:forward(output, label)
			local df_do = criterion:backward(output, label) -- 1x1x30x30  
			netD:backward(real_AB, df_do)
		end
	else
		m = real_C:float():resize(real_C:size(3), real_C:size(4))
		m = image.scale(m, output:size(3), output:size(4)):add(1):mul(0.5)
		m[m:gt(0)] = 1
		m:resize(1, 1, m:size(1), m:size(2))
		local layer1 = nn.CMul(output:size())
		layer1.weight = m:clone():mul(-1):add(1)
		local layer2 = nn.CAdd(output:size())
		layer2.bias = m:clone()
		if opt.gpu > 0 then
			layer1 = layer1:cuda()
			layer2 = layer2:cuda()
		end
		local netReal = nn.Sequential()
		netReal:add(layer1):add(layer2)
		local outputReal = netReal:forward(output)
		errD_real = criterion:forward(outputReal, label)
		local df_do = criterion:backward(outputReal, label)
		local df_dg = netReal:updateGradInput(output, df_do)
		netD:backward(real_AB, df_dg)
	 end
	 
	-- Fake
	local output = netD:forward(fake_AB) -- Subir el valor de la zona de la mascara
	label:fill(fake_label)

	if synth_label == 1 then
		if mGAN == 1 then
			m = real_C:float():resize(real_C:size(3), real_C:size(4))
			m = image.scale(m, output:size(3), output:size(4)):add(1):mul(0.5)
			m[m:gt(0)] = 1
			m:resize(1, 1, m:size(1), m:size(2))
			local layer1 = nn.CMul(output:size())
			local term_layer1 = m:clone():mul(opt.gamma-1):add(1)
			layer1.weight = term_layer1
			local layer2 = nn.Clamp(0, 1)
			if opt.gpu > 0 then
				layer1 = layer1:cuda()
				layer2 = layer2:cuda()
			end
			mGAN_D = nn.Sequential()
			mGAN_D:add(layer1):add(layer2)
			local output_mGAN = mGAN_D:forward(output)
			errD_fake = criterion:forward(output_mGAN, label)
			local df_do = criterion:backward(output_mGAN, label)
			local df_dg = mGAN_D:updateGradInput(output, df_do)
			netD:backward(fake_AB, df_dg)
		else
			errD_fake = criterion:forward(output, label)
			local df_do = criterion:backward(output, label)
			netD:backward(fake_AB, df_do)
		end
	else
		m = real_C:float():resize(real_C:size(3), real_C:size(4))
		m = image.scale(m, output:size(3), output:size(4)):add(1):mul(0.5)
		m[m:gt(0)] = 1
		m:resize(1, 1, m:size(1), m:size(2))
		local layer1 = nn.CMul(output:size())
		layer1.weight = m:clone():mul(-1):add(1)
		if opt.gpu > 0 then
			layer1 = layer1:cuda()
		end
		netReal = nn.Sequential()
		netReal:add(layer1)
		local outputReal = netReal:forward(output)
		errD_fake = criterion:forward(outputReal, label)
		local df_do = criterion:backward(outputReal, label)
		local df_dg = netReal:updateGradInput(output, df_do)
		netD:backward(real_AB, df_dg)
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
			output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
			local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
			if opt.gpu>0 then 
				label = label:cuda();
			end
			if mGAN == 1 then
				output_mGAN = mGAN_D.output
				errG = criterion:forward(output_mGAN, label)
				local df_do = criterion:backward(output_mGAN, label)
				local df_dx = mGAN_D:updateGradInput(output, df_do)
				df_dg = netD:updateGradInput(fake_AB, df_dx):narrow(2,fake_AB:size(2)-output_gan_nc+1, output_gan_nc)
			else
				errG = criterion:forward(output, label)
				local df_do = criterion:backward(output, label)
				df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
			end	   
		else
			errG = 0
		end
	else
		if opt.use_GAN==1 then
			output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
			local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
			if opt.gpu>0 then 
				label = label:cuda();
			end
			local outputReal = netReal.output
			errG = criterion:forward(outputReal, label)
			local df_do = criterion:backward(outputReal, label)
			local df_dx = netReal:updateGradInput(output, df_do)
			df_dg = netD:updateGradInput(fake_AB, df_dx):narrow(2,fake_AB:size(2)-output_gan_nc+1, output_gan_nc)
		else
			errG = 0
		end
	end
	 
	-- unary loss
	local df_dg_AE = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_dg_AE = df_dg_AE:cuda();
	end

	if synth_label == 1 then
		if opt.use_L1==1 then
			errL1 = criterionAE:forward(fake_B, realGray_B)
			df_dg_AE = criterionAE:backward(fake_B, realGray_B)
		else
			errL1 = 0
		end
	else   
		if opt.use_L1==1 then
			local m = real_C:clone():float()
			local layer1 = nn.CMul(m:size())
			layer1.weight = m:clone():mul(-1):add(1)
			local layer2 = nn.CAdd(m:size())
			layer2.bias = realGray_A:clone():float():cmul(m)
			if opt.gpu > 0 then
				layer1 = layer1:cuda()
				layer2 = layer2:cuda()
			end
			local netReal = nn.Sequential()
			netReal:add(layer1):add(layer2)
			local fake_B2 = netReal:forward(fake_B)
			errL1 = criterionAE:forward(fake_B2, realGray_B)
			local df_do_AE = criterionAE:backward(fake_B2, realGray_B)
			df_dg_AE = netReal:updateGradInput(fake_B, df_do_AE)
		else
			errL1 = 0
		end
	end
	
	netG:backward(realGray_A, df_dg + df_dg_AE:mul(opt.lambda))   

	return errG, gradParametersG
end

-- create closure to evaluate f(X) and df/dX of ss
local fSSx = function(x)
	gradParametersSS:zero()

	local df_dg = torch.zeros(fake_B:size())
	if opt.gpu>0 then 
		df_dg = df_dg:cuda();
	end
	local df_ddyn = torch.zeros(fake_C:size())
	if opt.gpu>0 then 
		df_ddyn = df_ddyn:cuda();
	end
	local df_derf = torch.zeros(erfnet_C:size())
	if opt.gpu>0 then 
		df_derf = df_derf:cuda();
	end
	local df_dy = torch.zeros(realRGB_A:size())
	if opt.gpu>0 then 
		df_dy = df_dy:cuda();
	end

	local output = netD.output -- last call of netD:forward{input_A,input_B} was already executed in fDx, so save computation (with the fake result)
	local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for SS cost
	if opt.gpu>0 then 
		label = label:cuda();
	end
	local outputReal = netReal.output
	errSS = criterion:forward(outputReal, label)
	local df_do = criterion:backward(outputReal, label)
	local df_dx = netReal:updateGradInput(output, df_do)
	df_dg = netD:updateGradInput(fake_AB, df_dx):narrow(2,fake_AB:size(2)-output_gan_nc+1, output_gan_nc)
	df_ddyn = netG:updateGradInput(fake_C, df_dg):narrow(2,fake_C:size(2)-output_gan_nc+1, output_gan_nc)
	df_derf = netDynSS:updateGradInput(erfnet_C, df_ddyn)
	
	fake_C = netSS.output
	errERFNet = criterionSS:forward(erfnet_C, real_C[1])
	df_dy = criterionSS:backward(erfnet_C, real_C[1])

	netSS:backward(realBGR_A, df_derf + df_dy:mul(opt.lambdaSS))

	return errSS, gradParametersSS
end

----------------------------------------------------------------------------

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

-- parse val_diplay_plot string into table
opt.val_display_plot = string.split(string.gsub(opt.val_display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.val_display_plot) do
	 if not util.containsValue({"val_errG", "val_errD", "val_errL1"}, v) then 
		  error(string.format('bad val_display_plot value "%s"', v)) 
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

----------------------------------------------------------------------------

-- main loop
local counter = 0
for epoch = 1, opt.niter do
	epoch_tm:reset()
	for i = 1, math.min(synth_data:size(), opt.ntrain), opt.batchSize do
		tm:reset()
		-- load a batch and run G on that batch
		if opt.NSYNTH_DATA_ROOT ~= '' and epoch > opt.epoch_synth then
			if torch.uniform() > opt.pNonSynth then
				synth_label = 1
			else
				synth_label = 0
			end
		end

		createRealFake()

		-- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
		if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end

		-- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
		optim.adam(fGx, parametersG, optimStateG)

		-- (3) Update SS network:
		if synth_label == 0 then optim.adam(fSSx, parametersSS, optimStateSS) end

		-- display
		counter = counter + 1
		if counter % opt.display_freq == 0 and opt.display then	
			createRealFake()
			local img_input = util.scaleBatch(realGray_A:float(),100,100):add(1):div(2)
			disp.image(img_input, {win=opt.display_id, title=opt.name .. ' input'})
			local mask_input = util.scaleBatch(real_C:float(),100,100):add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' mask'})
			local img_output = util.scaleBatch(fake_B:float(),100,100):add(1):div(2)
			disp.image(img_output, {win=opt.display_id+2, title=opt.name .. ' output'})
			local img_target = util.scaleBatch(realGray_B:float(),100,100):add(1):div(2)
			disp.image(img_target, {win=opt.display_id+3, title=opt.name .. ' target'})
			if synth_label == 0 then
				local mask_input = util.scaleBatch(real_C:float(),100,100):add(1):div(2)
				disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' targetMASK'})
				local dyn_mask_output = util.scaleBatch(fake_C:float(),100,100):add(1):div(2)
				disp.image(dyn_mask_output, {win=opt.display_id+4, title=opt.name .. ' dynamicMASK'})
				local mask_output = util.scaleBatch(winner:float(),100,100):add(1):div(2)
				disp.image(mask_output, {win=opt.display_id+5, title=opt.name .. ' outputMASK'})
			end
		end


		-- write display visualization to disk
		-- runs on the first batchSize images in the opt.phase set
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
				for i2=1, fake_B:size(1) do
					if image_out==nil then 
						image_out = torch.cat(realGray_A[i2]:float():add(1):div(2),
							fake_B[i2]:float():add(1):div(2),3)
					else
						image_out = torch.cat(image_out, torch.cat(realGray_A[i2]:float():add(1):div(2),
							fake_B[i2]:float():add(1):div(2),3), 2) 
					end
				end
			end
			image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
			opt.serial_batches=serial_batches
		end
		
		if (counter % opt.val_freq == 0 or counter == 1) and opt.display then
			val_createRealFake()
			val_errL1 = criterionAE:forward(val_fake_B, val_realGray_B)
			local img_input = util.scaleBatch(val_realGray_A:float(),100,100):add(1):div(2)
			disp.image(img_input, {win=opt.display_id+6, title=opt.name .. ' val_input'})
			local mask_input = util.scaleBatch(val_real_C:float(),100,100):add(1):div(2)
			disp.image(mask_input, {win=opt.display_id+7, title=opt.name .. ' val_mask'})
			local img_output = util.scaleBatch(val_fake_B:float(),100,100):add(1):div(2)
			disp.image(img_output, {win=opt.display_id+8, title=opt.name .. ' val_output'})
			local img_target = util.scaleBatch(val_realGray_B:float(),100,100):add(1):div(2)
			disp.image(img_target, {win=opt.display_id+9, title=opt.name .. ' val_target'})
			if synth_label == 0 then
				local mask_input = util.scaleBatch(val_real_C:float(),100,100):add(1):div(2)
				disp.image(mask_input, {win=opt.display_id+1, title=opt.name .. ' targetMASK'})
				local dyn_mask_output = util.scaleBatch(val_fake_C:float(),100,100):add(1):div(2)
				disp.image(dyn_mask_output, {win=opt.display_id+4, title=opt.name .. ' dynamicMASK'})
				local mask_output = util.scaleBatch(val_winner:float(),100,100):add(1):div(2)
				disp.image(mask_output, {win=opt.display_id+5, title=opt.name .. ' outputMASK'})
			end
		end

		-- logging and display plot
		if counter % opt.print_freq == 0 then
			local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1}
			local val_errG = nil
			local val_errD = nil
			local val_loss = {val_errG=val_errG and val_errG or -1, val_errD=val_errD and val_errD or -1, val_errL1=val_errL1 and val_errL1 or -1}
			local curItInBatch = ((i-1) / opt.batchSize)
			local totalItInBatch = math.floor(math.min(synth_data:size(), opt.ntrain) / opt.batchSize)
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
			if opt.NSYNTH_DATA_ROOT ~= '' then
				torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_SS.t7'), netSS:clearState())
			end
		end
	end

	parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
	parametersG, gradParametersG = nil, nil
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = nil, nil
	end
	
	if epoch % opt.save_epoch_freq == 0 then
		torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
		torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
		if opt.NSYNTH_DATA_ROOT ~= '' then
			torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_SS.t7'), netSS:clearState())
		end
	end

	print(('End of epoch %d / %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
	parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
	parametersG, gradParametersG = netG:getParameters()
	if opt.NSYNTH_DATA_ROOT ~= '' then
		parametersSS, gradParametersSS = netSS:getParameters()
	end
end