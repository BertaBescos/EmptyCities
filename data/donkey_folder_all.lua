
--[[
		This data loader is a modified version of the one from dcgan.torch
		(see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
		Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
		Copyright (c) 2015-present, Facebook, Inc.
		All rights reserved.
		This source code is licensed under the BSD-style license found in the
		LICENSE file in the root directory of this source tree. An additional grant
		of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
print(os.getenv('DATA_ROOT'))
opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)
print(opt.data)
paths.dofile('data_aug.lua')
-- This file contains the data augmentation techniques.

if not paths.dirp(opt.data) then
		error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local output_nc = opt.output_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local preprocess = function(im)
	if opt.data_aug == 1 then
		im = data_aug.apply(im)
	end


end

local preprocessAandB = function(imA, imB)

	if opt.data_aug == 1 then
			imA,imB,imB = data_aug.apply(imA,imB,imB)
	end

	imA = image.scale(imA, loadSize[2], loadSize[2])
	imB = image.scale(imB, loadSize[2], loadSize[2])

	local perm = torch.LongTensor{3, 2, 1}

	if input_nc == 3 then
		imA = imA:index(1, perm)--:mul(256.0): brg, rgb
	imB = imB:index(1, perm)
	end
	imA = imA:mul(2):add(-1)
	imB = imB:mul(2):add(-1)

	assert(imA:max()<=1,"A: badly scaled inputs")
	assert(imA:min()>=-1,"A: badly scaled inputs")
	assert(imB:max()<=1,"B: badly scaled inputs")
	assert(imB:min()>=-1,"B: badly scaled inputs")
 
	local oW = sampleSize[2]
	local oH = sampleSize[2]

	local iH = imA:size(2)
	local iW = imA:size(3)

	if iH~=oH then     
		h1 = math.ceil(torch.uniform(1e-2, iH-oH))
	end
	
	if iW~=oW then
		w1 = math.ceil(torch.uniform(1e-2, iW-oW))
	end
	if iH ~= oH or iW ~= oW then 
		imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
		imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
	end
	
	return imA, imB
end


--local function loadImage

local function loadImage(path)
	local input = image.load(path, input_nc, 'float')
	local h = input:size(2)
	local w = input:size(3)

	imA, imB, imC = nil

	if opt.mask == '' and opt.target == '' then
		imA = input
	end 
	if opt.mask == '' and opt.target ~= '' then
		imA = image.crop(input, 0, 0, w/2, h)
		imB = image.crop(input, w/2, 0, w, h)
	end
	if opt.mask ~= '' and opt.target == '' then
		imA = image.crop(input, 0, 0, w/2, h)
		imC = image.crop(input, w/2, 0, w, h)
	end
	if opt.mask ~= '' and opt.target ~= '' then
		imA = image.crop(input, 0, 0, w/3, h)
		imB = image.crop(input, w/3, 0, 2*w/3, h)
		imC = image.crop(input, 2*w/3, 0, w, h)
	end
	 return imA, imB, imC
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
	collectgarbage()
	local imA, imB, imC = loadImage(path)
	if imB ~= nil and imC ~= nil then
		imA, imB, imC = preprocessAandBandC(imA, imB, imC)
		im = torch.cat(imA, imB, 1)
		im = torch.cat(im, imC, 1)
	end
	if imB == nil and imC ~= nil then
		imA, imC = preprocessAandC(imA, imC)
		im = torch.cat(imA, imC, 1)
	end
	if imB ~= nil and imC == nil then
		imA, imB = preprocessAandB(imA, imB)
		im = torch.cat(imA, imB, 1)
	end
	if imB == nil and imC == nil then
		im = preprocess(imA)
	end

	print(im:size())
	print(im:type())
	return im
end

--------------------------------------
-- trainLoader
print('trainCache', trainCache)
--if paths.filep(trainCache) then
--   print('Loading train metadata from cache')
--   trainLoader = torch.load(trainCache)
--   trainLoader.sampleHookTrain = trainHook
--   trainLoader.loadSize = {input_nc, opt.loadSize, opt.loadSize}
--   trainLoader.sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]}
--   trainLoader.serial_batches = opt.serial_batches
--   trainLoader.split = 100
--else
print('Creating train metadata')
--   print(opt.data)
print('serial batch:, ', opt.serial_batches)
trainLoader = dataLoader{
		paths = {opt.data},
		loadSize = {input_nc, loadSize[2], loadSize[2]},
		sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]},
		split = 100,
		serial_batches = opt.serial_batches, 
		verbose = true
 }
--   print('finish')
--torch.save(trainCache, trainLoader)
--print('saved metadata cache at', trainCache)
trainLoader.sampleHookTrain = trainHook
--end
collectgarbage()

-- do some sanity checks on trainLoader
do
	 local class = trainLoader.imageClass
	 local nClasses = #trainLoader.classes
	 assert(class:max() <= nClasses, "class logic has error")
	 assert(class:min() >= 1, "class logic has error")
end
