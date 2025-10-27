static const char *io_port_to_string(uint32_t io_port)

{

    if (io_port >= QXL_IO_RANGE_SIZE) {

        return "out of range";

    }

    static const char *io_port_to_string[QXL_IO_RANGE_SIZE + 1] = {

        [QXL_IO_NOTIFY_CMD]             = "QXL_IO_NOTIFY_CMD",

        [QXL_IO_NOTIFY_CURSOR]          = "QXL_IO_NOTIFY_CURSOR",

        [QXL_IO_UPDATE_AREA]            = "QXL_IO_UPDATE_AREA",

        [QXL_IO_UPDATE_IRQ]             = "QXL_IO_UPDATE_IRQ",

        [QXL_IO_NOTIFY_OOM]             = "QXL_IO_NOTIFY_OOM",

        [QXL_IO_RESET]                  = "QXL_IO_RESET",

        [QXL_IO_SET_MODE]               = "QXL_IO_SET_MODE",

        [QXL_IO_LOG]                    = "QXL_IO_LOG",

        [QXL_IO_MEMSLOT_ADD]            = "QXL_IO_MEMSLOT_ADD",

        [QXL_IO_MEMSLOT_DEL]            = "QXL_IO_MEMSLOT_DEL",

        [QXL_IO_DETACH_PRIMARY]         = "QXL_IO_DETACH_PRIMARY",

        [QXL_IO_ATTACH_PRIMARY]         = "QXL_IO_ATTACH_PRIMARY",

        [QXL_IO_CREATE_PRIMARY]         = "QXL_IO_CREATE_PRIMARY",

        [QXL_IO_DESTROY_PRIMARY]        = "QXL_IO_DESTROY_PRIMARY",

        [QXL_IO_DESTROY_SURFACE_WAIT]   = "QXL_IO_DESTROY_SURFACE_WAIT",

        [QXL_IO_DESTROY_ALL_SURFACES]   = "QXL_IO_DESTROY_ALL_SURFACES",

#if SPICE_INTERFACE_QXL_MINOR >= 1

        [QXL_IO_UPDATE_AREA_ASYNC]      = "QXL_IO_UPDATE_AREA_ASYNC",

        [QXL_IO_MEMSLOT_ADD_ASYNC]      = "QXL_IO_MEMSLOT_ADD_ASYNC",

        [QXL_IO_CREATE_PRIMARY_ASYNC]   = "QXL_IO_CREATE_PRIMARY_ASYNC",

        [QXL_IO_DESTROY_PRIMARY_ASYNC]  = "QXL_IO_DESTROY_PRIMARY_ASYNC",

        [QXL_IO_DESTROY_SURFACE_ASYNC]  = "QXL_IO_DESTROY_SURFACE_ASYNC",

        [QXL_IO_DESTROY_ALL_SURFACES_ASYNC]

                                        = "QXL_IO_DESTROY_ALL_SURFACES_ASYNC",

        [QXL_IO_FLUSH_SURFACES_ASYNC]   = "QXL_IO_FLUSH_SURFACES_ASYNC",

        [QXL_IO_FLUSH_RELEASE]          = "QXL_IO_FLUSH_RELEASE",

#endif

    };

    return io_port_to_string[io_port];

}
