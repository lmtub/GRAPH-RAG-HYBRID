static int commit_direntries(BDRVVVFATState* s,

	int dir_index, int parent_mapping_index)

{

    direntry_t* direntry = array_get(&(s->directory), dir_index);

    uint32_t first_cluster = dir_index == 0 ? 0 : begin_of_direntry(direntry);

    mapping_t* mapping = find_mapping_for_cluster(s, first_cluster);



    int factor = 0x10 * s->sectors_per_cluster;

    int old_cluster_count, new_cluster_count;

    int current_dir_index = mapping->info.dir.first_dir_index;

    int first_dir_index = current_dir_index;

    int ret, i;

    uint32_t c;



DLOG(fprintf(stderr, "commit_direntries for %s, parent_mapping_index %d\n", mapping->path, parent_mapping_index));



    assert(direntry);

    assert(mapping);

    assert(mapping->begin == first_cluster);

    assert(mapping->info.dir.first_dir_index < s->directory.next);

    assert(mapping->mode & MODE_DIRECTORY);

    assert(dir_index == 0 || is_directory(direntry));



    mapping->info.dir.parent_mapping_index = parent_mapping_index;



    if (first_cluster == 0) {

	old_cluster_count = new_cluster_count =

	    s->last_cluster_of_root_directory;

    } else {

	for (old_cluster_count = 0, c = first_cluster; !fat_eof(s, c);

		c = fat_get(s, c))

	    old_cluster_count++;



	for (new_cluster_count = 0, c = first_cluster; !fat_eof(s, c);

		c = modified_fat_get(s, c))

	    new_cluster_count++;

    }



    if (new_cluster_count > old_cluster_count) {

	if (insert_direntries(s,

		current_dir_index + factor * old_cluster_count,

		factor * (new_cluster_count - old_cluster_count)) == NULL)

	    return -1;

    } else if (new_cluster_count < old_cluster_count)

	remove_direntries(s,

		current_dir_index + factor * new_cluster_count,

		factor * (old_cluster_count - new_cluster_count));



    for (c = first_cluster; !fat_eof(s, c); c = modified_fat_get(s, c)) {

	void* direntry = array_get(&(s->directory), current_dir_index);

	int ret = vvfat_read(s->bs, cluster2sector(s, c), direntry,

		s->sectors_per_cluster);

	if (ret)

	    return ret;

	assert(!strncmp(s->directory.pointer, "QEMU", 4));

	current_dir_index += factor;

    }



    ret = commit_mappings(s, first_cluster, dir_index);

    if (ret)

	return ret;



    /* recurse */

    for (i = 0; i < factor * new_cluster_count; i++) {

	direntry = array_get(&(s->directory), first_dir_index + i);

	if (is_directory(direntry) && !is_dot(direntry)) {

	    mapping = find_mapping_for_cluster(s, first_cluster);

	    assert(mapping->mode & MODE_DIRECTORY);

	    ret = commit_direntries(s, first_dir_index + i,

		array_index(&(s->mapping), mapping));

	    if (ret)

		return ret;

	}

    }



    return 0;

}
