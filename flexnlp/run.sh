echo "...producing sample assembly and data......\n"
python3 produce_test.py
echo "\n...generating flexnlp-ila program fragment...\n"
python3 gen_prog_frag.py asm_test.json data_lib_test.json prog_frag_test_in.json
