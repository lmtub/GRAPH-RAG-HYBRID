opts_visitor_new(const QemuOpts *opts)

{

    OptsVisitor *ov;




    ov = g_malloc0(sizeof *ov);



    ov->visitor.type = VISITOR_INPUT;



    ov->visitor.start_struct = &opts_start_struct;

    ov->visitor.check_struct = &opts_check_struct;

    ov->visitor.end_struct   = &opts_end_struct;



    ov->visitor.start_list = &opts_start_list;

    ov->visitor.next_list  = &opts_next_list;

    ov->visitor.end_list   = &opts_end_list;



    ov->visitor.type_int64  = &opts_type_int64;

    ov->visitor.type_uint64 = &opts_type_uint64;

    ov->visitor.type_size   = &opts_type_size;

    ov->visitor.type_bool   = &opts_type_bool;

    ov->visitor.type_str    = &opts_type_str;



    /* type_number() is not filled in, but this is not the first visitor to

     * skip some mandatory methods... */



    ov->visitor.optional = &opts_optional;

    ov->visitor.free = opts_free;



    ov->opts_root = opts;



    return &ov->visitor;

}