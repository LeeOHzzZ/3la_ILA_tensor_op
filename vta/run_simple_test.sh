echo "... generating simple test assembly JSON and data mapping JSON ... \n"
python3 produce_test.py
echo "\n... generating ILA program fragment .... \n"
python3 produce_prog_frag.py asm_test.json data_lib_test.json ./prog_frag/prog_frag_test_in.json
