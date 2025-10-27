void tcg_context_init(TCGContext *s)

{

    int op, total_args, n;

    TCGOpDef *def;

    TCGArgConstraint *args_ct;

    int *sorted_args;



    memset(s, 0, sizeof(*s));

    s->nb_globals = 0;

    

    /* Count total number of arguments and allocate the corresponding

       space */

    total_args = 0;

    for(op = 0; op < NB_OPS; op++) {

        def = &tcg_op_defs[op];

        n = def->nb_iargs + def->nb_oargs;

        total_args += n;

    }



    args_ct = g_malloc(sizeof(TCGArgConstraint) * total_args);

    sorted_args = g_malloc(sizeof(int) * total_args);



    for(op = 0; op < NB_OPS; op++) {

        def = &tcg_op_defs[op];

        def->args_ct = args_ct;

        def->sorted_args = sorted_args;

        n = def->nb_iargs + def->nb_oargs;

        sorted_args += n;

        args_ct += n;

    }



    /* Register helpers.  */

#define GEN_HELPER 2

#include "helper.h"



    tcg_target_init(s);

}
