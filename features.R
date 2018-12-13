library(flacco)
files = list.files()
features = NULL
for (file in files) {
  dat = read.csv(file)
  dat[1] = NULL
  num_cols = dim(dat)[2]
  inputs = dat[1:(num_cols-2)]
  outputs = dat[num_cols]
  inputs = apply(inputs, 2 , as.numeric)
  outputs = apply(outputs, 1, as.numeric)
  feat.object = createFeatureObject(X = inputs, y = outputs)
  features = rbind(features, data.frame(calculateFeatureSet(feat.object, set = "ela_meta")))
}
write.csv(files, file = 'filenames.csv')
write.csv(features, file = 'features.csv')