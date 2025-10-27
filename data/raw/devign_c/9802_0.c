void spapr_register_hypercall(target_ulong opcode, spapr_hcall_fn fn)

{

    spapr_hcall_fn *slot;



    if (opcode <= MAX_HCALL_OPCODE) {

        assert((opcode & 0x3) == 0);



        slot = &papr_hypercall_table[opcode / 4];

    } else {

        assert((opcode >= KVMPPC_HCALL_BASE) && (opcode <= KVMPPC_HCALL_MAX));





        slot = &kvmppc_hypercall_table[opcode - KVMPPC_HCALL_BASE];

    }



    assert(!(*slot) || (fn == *slot));

    *slot = fn;

}
