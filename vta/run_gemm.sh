echo "...generating gemm test inputs: blocked_gemm_asm.JSON and data mapping JSON file....\n"
cd test_gen
./gen_block_gemm_test.sh
cd ..
echo "\n...generating ILA program fragment....\n"
python3 produce_prog_frag.py test_gen/asm/blocked_gemm_asm.json test_gen/data_dump/Blocked_GEMM_test-batch=16-channels=16-block=16-uop_comp=0-vt=1_data_dump.json prog_frag/blocked_gemm_input.json
