World *world_alloc(Rocker *r, size_t sizeof_private,

                   enum rocker_world_type type, WorldOps *ops)

{

    World *w = g_malloc0(sizeof(World) + sizeof_private);



    if (w) {

        w->r = r;

        w->type = type;

        w->ops = ops;

        if (w->ops->init) {

            w->ops->init(w);

        }

    }



    return w;

}
