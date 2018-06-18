

trainMean = torch.FloatTensor(3)
trainStd = torch.FloatTensor(3)
local meanEstimate = {0,0,0}
local stdEstimate = {0,0,0}
histClasses = torch.FloatTensor(#classes):zero()
histClassesEncoder = torch.FloatTensor(#classes):zero()

print '==> Calculating dataset mean/std and class balances (cache data)'
for i = 1,trsize do
	local imgPath = trainData.data[i]
        local gtPath = trainData.labels[i]

        local dataTemp = image.load(imgPath)--gm.load(imgPath, 'byte')
        local img = image.scale(dataTemp,opt.imWidth, opt.imHeight)

        for j=1,3 do
           meanEstimate[j] = meanEstimate[j] + img[j]:mean()
           stdEstimate[j] = stdEstimate[j] + img[j]:std()
        end

        -- label image data are resized to be [1,nClasses] in [0 255] scale:
	print(gtPath)
        local labelIn = image.load(gtPath, 1, 'byte') --gm.load(gtPath, 'byte')[1]
        local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

        labelFile:apply(function(x) return classMap[x][1] end)

        local labelFileEncoder = image.scale(labelFile, opt.labelWidth, opt.labelHeight, 'simple'):float()

        histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)
        histClassesEncoder = histClassesEncoder + torch.histc(labelFileEncoder, #classes, 1, #classes)

        xlua.progress(i, trsize)
end
