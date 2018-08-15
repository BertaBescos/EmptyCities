-- usage: DATA_ROOT=/path/to/data/ th test.lua
-- usage: input=/path/to/input/image/ mask=/path/to/mask/ target=/path/to/target/image/ th test.lua
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    DATA_ROOT = '',             -- path to images (should have subfolders 'train', 'val', etc)
    input = '',                 
    mask = '',
    target = '',
    batchSize = 1,              -- # images in batch
    loadSize = 256,             -- scale images to this size
    fineSize = 256,             --  then crop to this size
    display = 1,                -- display samples while training. 0 = false
    display_id = 200,           -- display window id.
    gpu = 1,                    -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    phase = 'val',              -- train, val, test ,etc
    aspect_ratio = 1.0,         -- aspect ratio of result images
    name = 'mGAN',              -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,               -- #  of input image channels
    output_nc = 3,              -- #  of output image channels
    serial_batches = 1,         -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,      -- iter into serial image list
    cudnn = 1,                  -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints',  -- loads models from here
    results_dir='./results/',   -- saves results here
    which_epoch = 'latest',     -- which epoch to test? set to 'latest' to use latest cached model
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
local netSS_name = 'SemSeg/erfnet.net'

-- useful function for debugging
function pause ()
    print("Press any key to continue.")
    io.flush()
    io.read()
end

data_loader = paths.dofile('data/dataINF.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

-- index different inputs
local idx_A = nil
local idx_B = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc
idx_A = {1, input_nc}
idx_B = {input_nc+1, input_nc+output_nc}

----------------------------------------------------------------------------

local inputRGB = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
if opt.target ~= '' then
    targetRGB = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
end
if opt.mask ~= '' then
    inputMask = torch.FloatTensor(opt.batchSize,1,opt.fineSize,opt.fineSize)
end

-- load all models
print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
netG:evaluate()
print(netG)
if opt.mask ~= '' then
    netSS = torch.load(paths.concat(opt.checkpoints_dir, netSS_name))
    netSS:evaluate()
    print(netSS)
    netDynSS = nn.Sequential()
    local convDyn = nn.SpatialFullConvolution(20,1,1,1,1,1)
    local w, dw = convDyn:parameters()
    w[1][{{1,12}}] = -8/20 -- Static
    w[1][{{13,20}}] = 12/20 -- Dynamic
    w[2]:fill(0)
    netDynSS:add(nn.SoftMax())
    netDynSS:add(convDyn):add(nn.MulConstant(100))
    netDynSS:add(nn.Tanh())
    netDynSS = netDynSS:cuda()
    print(netDynSS)
end

-- this function will be used later for website
function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

local filepaths = {} -- paths to images tested on

for n=1,math.floor(data:size()/opt.batchSize) do
    print('processing batch ' .. n)
    
    local data_curr, filepaths_curr = data:getBatch()
    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    inputRGB = data_curr[{ {}, idx_A, {}, {} }]
    local inputGray = image.rgb2y(inputRGB[1])
    if opt.gpu > 0 then
        inputRGB = inputRGB:cuda()
        inputGray = inputGray:cuda()
    end
    inputGray = inputGray:resize(1,inputGray:size(1),inputGray:size(2),inputGray:size(3))
    local inputBGR = inputRGB:clone()
    inputBGR = inputBGR:add(1):mul(0.5)
    inputBGR[1][1] = inputRGB[1][3]:add(1):mul(0.5)
    inputBGR[1][3] = inputRGB[1][1]:add(1):mul(0.5)

    if opt.target ~= '' then
        targetRGB = data_curr[{ {}, idx_B, {}, {} }]
        targetGray = image.rgb2y(targetRGB[1])
    end

    if opt.mask ~= '' then
        idx_C = {input_nc + output_nc + 1,input_nc + output_nc + 1}
        inputMask = data_curr[{ {}, idx_C, {}, {} }]
        inputMask = inputMask:cuda()
        inputGAN = torch.cat(inputGray,inputMask,2)
    else
        inputMask = netSS:forward(inputBGR)
        inputMask = netDynSS:forward(inputMask)
        inputGAN = torch.cat(inputGray,inputMask,2)
    end

    output = netG:forward(inputGAN)

    input = inputGray:float():add(1):div(2)
    output = output:float():add(1):div(2)
    target = targetGray:add(1):div(2)

    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))
    if opt.target ~= '' then
        paths.mkdir(paths.concat(image_dir,'target'))
    end

    for i=1, opt.batchSize do
        image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
    end

    if opt.target ~= '' then
        for i=1, opt.batchSize do
            image.save(paths.concat(image_dir,'target',filepaths_curr[i]), image.scale(target[i],target[i]:size(2),target[i]:size(3)/opt.aspect_ratio))
        end
    end

    print('Saved images to: ', image_dir)

    filepaths = TableConcat(filepaths, filepaths_curr)

    if opt.display then
        disp = require 'display'
        disp.image(util.scaleBatch(input,100,100),{win=opt.display_id, title='input'})
        disp.image(util.scaleBatch(output,100,100),{win=opt.display_id+1, title='output'})
        disp.image(util.scaleBatch(target,100,100),{win=opt.display_id+2, title='target'})
        
        print('Displayed images')
    end
    
    filepaths = TableConcat(filepaths, filepaths_curr)
end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))

io.write('<table style="text-align:center;">')

if opt.target ~= '' then
    io.write('<tr><td>Image #</td><td>Input</td><td>Output</td><td>Ground Truth</td></tr>')
    for i=1, #filepaths do
        io.write('<tr>')
        io.write('<td>' .. filepaths[i] .. '</td>')
        io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
        io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
        io.write('<td><img src="./images/target/' .. filepaths[i] .. '"/></td>')
        io.write('</tr>')
    end
else
    io.write('<tr><td>Image #</td><td>Input</td><td>Output</td></tr>')
    for i=1, #filepaths do
        io.write('<tr>')
        io.write('<td>' .. filepaths[i] .. '</td>')
        io.write('<td><img src="./images/input/' .. filepaths[i] .. '"/></td>')
        io.write('<td><img src="./images/output/' .. filepaths[i] .. '"/></td>')
        io.write('</tr>')
    end
end

io.write('</table>')