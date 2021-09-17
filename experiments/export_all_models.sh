mkdir -p exported_models

echo "Export covid-if model"
cd covid-if
python export_bioimageio_model.py -c checkpoints/covid-if-affinity-model -i /scratch/pape/covid-if/gt_image_000.h5 -o ../exported_models/covid_if_boundaries -a 1 -f torchscript
cd ..
 
echo "Export cremi model"
cd neuron-segmentation/cremi
python export_bioimageio_model.py -c checkpoints/affinity_model_default -i /scratch/pape/cremi/sample_C_20160501.hdf -o ../../exported_models/cremi_boundaries -a 1 -f torchscript
cd ../..
 
echo "Export dsb model"
cd dsb
python export_bioimageio_model.py -c checkpoints/dsb-affinity-model -i /scratch/pape/dsb/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif -o ../exported_models/dsb_boundaries -a 1 -f torchscript
cd ..

echo "Export monuseg model"
cd monuseg
python export_bioimageio_model.py -c checkpoints/monuseg-affinity-model -i /g/kreshuk/pape/Work/data/monuseg/images/TCGA-18-5592-01Z-00-DX1.tif -o ../exported_models/monuseg_boundaries -a 1 -f torchscript
cd ..

echo "Export isbi2012 model"
cd neuron-segmentation/isbi2012
python export_bioimageio_model.py -c checkpoints/affinity-model -i /g/kreshuk/data/isbi2012_challenge/isbi2012_test_volume.h5 -o ../../exported_models/isbi2012_boundaries -a 1 -f torchscript
cd ../..

echo "Export mito-em model"
cd mito-em
python export_bioimageio_model.py -c checkpoints/affinity_model_default_human_rat -i /scratch/pape/mito_em/data/human_test.n5 -o ../exported_models/mito_em_boundaries -a 1 -f torchscript
cd ..
 
echo "Export plantseg models"
cd plantseg
python export_bioimageio_model.py -c ovules/checkpoints/affinity_model2d -i /g/kreshuk/wolny/Datasets/Ovules/GT2x/val/N_420_ds2x.h5 -o ../exported_models/ovules_boundaries -a 1 -f torchscript
cd ..
 
echo "Export platynereis models"
cd platynereis

echo "Export cells"
python export_bioimageio_model.py -c cells/checkpoints/affinity_model -i /scratch/pape/platy/membrane/train_data_membrane_09.n5 -o ../exported_models/platy-cell-boundaries -a 1 -f torchscript

echo "Export nuclei"
python export_bioimageio_model.py -c nuclei/checkpoints/affinity_model -i /scratch/pape/platy/nuclei/train_data_nuclei_12.h5 -o ../exported_models/platy-nucleus-boundaries -a 1 -f torchscript

cd ..
