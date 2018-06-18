-- usage: DATA_ROOT=/path/to/data/ name=expt1 which_direction=BtoA th test.lua
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    DATA_ROOT = '',           -- path to images (should have subfolders 'train', 'val', etc)
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    display = 1,              -- display samples while training. 0 = false
    display_id = 200,         -- display window id.
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
    which_direction = 'AtoB', -- AtoB or BtoA
    phase = 'val',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    input_mask_nc = 0,	      -- #  of input mask channels --bbescos
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'

local input_mask_nc = opt.input_mask_nc

data_loader = paths.dofile('data/dataINF.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- translation direction
local idx_A = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc

idx_A = {1, input_nc}

----------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
--netG:evaluate()

print(netG)

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

if opt.how_many=='all' then
    opt.how_many=data:size()
end
opt.how_many=math.min(opt.how_many, data:size())

local filepaths = {} -- paths to images tested on
for n=1,math.floor(opt.how_many/opt.batchSize) do
    print('processing batch ' .. n)
    
    local data_curr, filepaths_curr = data:getBatch()
    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    input = data_curr[{ {}, idx_A, {}, {} }]

    if opt.gpu > 0 then
        input = input:cuda()
    end

    if input_mask_nc == 1 then
	idx_C = {input_nc + 1,input_nc + input_mask_nc}
	aux = torch.Tensor(opt.batchSize, input_mask_nc, opt.fineSize, opt.fineSize)
	aux = aux:cuda()
	aux:copy(data_curr[{ {}, idx_C, {}, {} }])
	input = torch.cat(input,aux,2)
    end

    if input_nc == 3 and input_mask_nc == 0 then  
	output = util.deprocess_batch(netG:forward(input))
	input = util.deprocess_batch(input):float()
	output = output:float()
    end
    if input_nc == 1 and input_mask_nc == 0 then  
	local _output = netG:forward(input)
	output = _output:add(1):div(2)
	output = output:float()
	input = input:add(1):div(2):float()
    end
    if input_nc == 3 and input_mask_nc == 1 then  
	output = util.deprocess_batch(netG:forward(input))
	local _input = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
	_input = _input:cuda()
	_input:copy(data_curr[{ {}, idx_A, {}, {} }])
        input = util.deprocess_batch(_input):float()
        output = output:float()
    end
    if input_nc == 1 and input_mask_nc == 1 then  
	local _input = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
	_input = _input:cuda()
	_input:copy(data_curr[{ {}, idx_A, {}, {} }])
	local _output = netG:forward(input)
	input = _input:add(1):div(2):float()
	output = _output:add(1):div(2)
	output = output:float()
    end

    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))

    for i=1, opt.batchSize do
        image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
    end

    print('Saved images to: ', image_dir)

    filepaths = TableConcat(filepaths, filepaths_curr)

end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))

io.write('<table style="text-align:center;">')

io.write('<tr><td>Image #</td><td>Input</td><td>Output</td></tr>')
for i=1, #filepaths do
    io.write('<tr>')
    io.write('<td>' .. filepaths[i] .. '</td>')
    io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
    io.write('</tr>')
end

io.write('</table>')
