static void vma_delete(struct mm_struct *mm)

{

    struct vm_area_struct *vma;



    while ((vma = vma_first(mm)) != NULL) {

        TAILQ_REMOVE(&mm->mm_mmap, vma, vma_link);

        qemu_free(vma);

    }

    qemu_free(mm);

}
