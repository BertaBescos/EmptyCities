
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
print(os.getenv('NSYNTH_DATA_ROOT'))
opt.data = paths.concat(os.getenv('NSYNTH_DATA_ROOT'), opt.phase)

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
local mask_nc = opt.input_mask_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local preprocessAandBandC = function(imA, imB, imC)

  if opt.rotation == 1 and torch.uniform() > 0.5 then 
    deg = torch.uniform(-30,30)

    imA = image.rotate(imA, deg * math.pi / 180, 'bilinear')
    imB = image.rotate(imB, deg * math.pi / 180, 'bilinear')
    imC = image.rotate(imC, deg * math.pi / 180, 'bilinear')

    deg = math.abs(deg)
    h = imA:size(2)
    w = imA:size(3)
    bb_w = w * math.cos(deg * math.pi / 180) + h * math.sin(deg * math.pi / 180)
    bb_h = w * math.sin(deg * math.pi / 180) + h * math.cos(deg * math.pi / 180)

    gamma = math.atan2(bb_w, bb_w)
    delta = math.pi - deg * math.pi / 180 - gamma

    d = w * math.cos(deg * math.pi / 180)
    a = d * math.sin(deg * math.pi / 180) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    w_new = bb_w - 2 * x
    h_new = bb_h - 2 * y

    imA = image.crop(imA, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)
    imB = image.crop(imB, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)
    imC = image.crop(imC, (w-w_new)/2, (h-h_new)/2, w_new + (w-w_new)/2, h_new + (h-h_new)/2)

  end

  imA = image.scale(imA, loadSize[2], loadSize[2])
  imB = image.scale(imB, loadSize[2], loadSize[2])
  imC = image.scale(imC, loadSize[2], loadSize[2]) -- between 0 and 1
  imC[imC:gt(0)] = 1

  local perm = torch.LongTensor{3, 2, 1}
  if input_nc == 3 then
  	imA = imA:index(1, perm)--:mul(256.0): brg, rgb
  	imB = imB:index(1, perm)
  end
  
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
    imC = image.crop(imC, w1, h1, w1 + oW, h1 + oH)
  end
  
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
    imC = image.hflip(imC)
  end

  if opt.gaussian_blur == 1 and torch.uniform() > 0.5 then
    local width = math.ceil(torch.uniform(0,6))
    local height = width
    local sigma_horz=torch.uniform(0.05, 0.25)
    local sigma_vert = sigma_horz
    local gs = image.gaussian{amplitude=1, 
                                normalize=true, 
                                width=width, 
                                height=height, 
                                sigma_horz=sigma_horz, 
                                sigma_vert=sigma_vert}
    imA = image.convolve(imA,gs,'same')
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
  end

  if opt.gaussian_noise == 1 and torch.uniform() > 0.5 then
    local augNoise = torch.uniform(0,0.1)
    local gs = torch.randn(imA:size()):float()*augNoise
    imA = torch.add(imA, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
  end

  if opt.brightness == 1 and torch.uniform() > 0.5 then
    local gs
    local var = 1
    gs = gs or imA.new()
    gs:resizeAs(imA):zero()
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
  end

  if opt.contrast == 1 and torch.uniform() > 0.5 then
    local gs
    local var = 1
    gs = gs or imA.new()
    gs = imA:clone()
    gs:fill(gs:mean())
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
  end

  if opt.saturation == 1 and torch.uniform() > 0.5 then
    local gs
    local var = 1
    gs = gs or imA.new()
    gs = imA:clone()
    local alpha = 1.0 + torch.uniform(-var, var)
    imA = imA:mul(alpha):add(1 - alpha, gs)
    imA = imA:add(-imA:min())
    imA = imA:div(imA:max())
  end

  if opt.dropout == 1 and torch.uniform() > 0.5 then
    local height = math.ceil(torch.uniform(10, 80))
    local width = math.ceil(torch.uniform(10, 80))
    h1 = math.ceil(torch.uniform(1e-2, imA:size(2)-height))
    w1 = math.ceil(torch.uniform(1e-2, imA:size(3)-width))
    imA[{{1},{h1,h1+height},{w1,w1+width}}] = 0
    imB[{{1},{h1,h1+height},{w1,w1+width}}] = 0
  end

  imA = imA:mul(2):add(-1)
  imB = imB:mul(2):add(-1)
  imC = imC:mul(2):add(-1) -- min(imC) = -1 & max(imC) = 1
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")
  assert(imC:max()<=1,"C: badly scaled inputs")
  assert(imC:min()>=-1,"C: badly scaled inputs")

  return imA, imB, imC
end

local function loadImageChannel(path)
    local input = image.load(path, 3, 'float')
    input = image.scale(input, loadSize[2], loadSize[2])

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = input:size(2)
    local iW = input:size(3)
    
    if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end
    
    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    end
    
    if opt.flip == 1 and torch.uniform() > 0.5 then 
      input = image.hflip(input)
    end
    
--    print(input:mean(), input:min(), input:max())
    local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()
    local imA = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0
    local imB = input_lab[{{2,3},{},{}}]:div(110.0)
    local imAB = torch.cat(imA, imB, 1)
    assert(imAB:max()<=1,"A: badly scaled inputs")
    assert(imAB:min()>=-1,"A: badly scaled inputs")
    
    return imAB
end

--local function loadImage

local function loadImage(path)
   local input = image.load(path, input_nc, 'float')

   if input_nc == 1 then
   	input:resize(1,input:size(1),input:size(2))
   end 	

   local h = input:size(2)
   local w = input:size(3)

   local imA = image.crop(input, 0, 0, w/2, h)
   local imB = imA:clone()
   local imC = image.crop(input, w/2, 0, w, h)

   return imA, imB, imC
end

local function loadImageInpaint(path)
  local imB = image.load(path, 3, 'float')
  imB = image.scale(imB, loadSize[2], loadSize[2])
  local perm = torch.LongTensor{3, 2, 1}
  imB = imB:index(1, perm)--:mul(256.0): brg, rgb
  imB = imB:mul(2):add(-1)
  assert(imB:max()<=1,"A: badly scaled inputs")
  assert(imB:min()>=-1,"A: badly scaled inputs")
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imB:size(2)
  local iW = imB:size(3)
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end
  local imA = imB:clone()
  imA[{{},{1 + oH/4, oH/2 + oH/4},{1 + oW/4, oW/2 + oW/4}}] = 1.0
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end
  imAB = torch.cat(imA, imB, 1)
  return imAB
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   if opt.preprocess == 'regular' then
--     print('process regular')
     local imA, imB, imC = loadImage(path)
     imA, imB, imC = preprocessAandBandC(imA, imB, imC)
     imAB = torch.cat(imA, imB, 1)
     imABC = torch.cat(imAB, imC, 1)
     -- print(imABC:size())
   end
   
   if opt.preprocess == 'colorization' then 
--     print('process colorization')
     imABC = loadImageChannel(path)
   end

   if opt.preprocess == 'inpaint' then
    -- print('process inpaint')
     imABC = loadImageInpaint(path)  
   end
  -- print('image AB size')
  -- print(imAB:size())
   return imABC
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
    sampleSize = {input_nc+output_nc+mask_nc, sampleSize[2], sampleSize[2]},
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
