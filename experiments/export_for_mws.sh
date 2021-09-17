mkdir -p exported_models_mws

echo "Export dsb model"
cd dsb
python export_bioimageio_model.py -c checkpoints/dsb-affinity-model -i /scratch/pape/dsb/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif -o ../exported_models_mws/dsb -a 0 -f torchscript onnx
cd ..

echo "Export isbi2012 model"
cd neuron-segmentation/isbi2012
python export_bioimageio_model.py -c checkpoints/affinity-model -i /g/kreshuk/data/isbi2012_challenge/isbi2012_test_volume.h5 -o ../../exported_models_mws/isbi2012 -a 0 -f torchscript onnx
cd ../..
