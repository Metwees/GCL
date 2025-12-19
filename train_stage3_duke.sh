CUDA_VISIBLE_DEVICES=0 python examples/main.py \
--name duke_init_JVTC_unsupervised \
--resume \
--stage 3 \
--epochs 20 \
--dataset-target dukemtmc-reid \
--mesh-dir ./examples/mesh/DukeMTMC/ \
--rho 0.002 \
--k1 30


CUDA_VISIBLE_DEVICES=0 python examples/main.py --name duke_init_JVTC_unsupervised --resume --stage 3 --epochs 20 --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --rho 0.002 --k1 30
python examples/generate_data.py --name duke_init_JVTC_unsupervised --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 45 --idnet-fix

python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 45 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 90 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 135 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 180 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 225 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 270 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/generate_data.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --degree 315 --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv


#python examples/tta.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --mode nv
python examples/tta.py --dataset-target dukemtmc-reid --mesh-dir ./examples/mesh/DukeMTMC/ --idnet-fix --init ./outputs/duke_init_JVTC_unsupervised/checkpoints/id_00070000.pt --gen 0 --b 8 --workers 0 --lf
