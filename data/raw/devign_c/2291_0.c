static void kqemu_record_pc(unsigned long pc)

{

    unsigned long h;

    PCRecord **pr, *r;



    h = pc / PC_REC_SIZE;

    h = h ^ (h >> PC_REC_HASH_BITS);

    h &= (PC_REC_HASH_SIZE - 1);

    pr = &pc_rec_hash[h];

    for(;;) {

        r = *pr;

        if (r == NULL)

            break;

        if (r->pc == pc) {

            r->count++;

            return;

        }

        pr = &r->next;

    }

    r = malloc(sizeof(PCRecord));

    r->count = 1;

    r->pc = pc;

    r->next = NULL;

    *pr = r;

    nb_pc_records++;

}
