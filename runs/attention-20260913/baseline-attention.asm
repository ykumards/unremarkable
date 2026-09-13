0000f4e4 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)>:
    f4e4: e92d 4ff0    	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
    f4e8: 4606         	mov	r6, r0
    f4ea: ed2d 8b02    	vpush	{d8}
    f4ee: b08f         	sub	sp, #0x3c
    f4f0: 468a         	mov	r10, r1
    f4f2: 4693         	mov	r11, r2
    f4f4: 9d1a         	ldr	r5, [sp, #0x68]
    f4f6: e9dd 941b    	ldrd	r9, r4, [sp, #108]
    f4fa: 900a         	str	r0, [sp, #0x28]
    f4fc: 4629         	mov	r1, r5
    f4fe: fb09 f805    	mul	r8, r9, r5
    f502: 9308         	str	r3, [sp, #0x20]
    f504: 461d         	mov	r5, r3
    f506: 4618         	mov	r0, r3
    f508: f07b ffda    	bl	0x8b4c0 <__divsi3>      @ imm = #0x7bfb4
    f50c: 9006         	str	r0, [sp, #0x18]
    f50e: 2d00         	cmp	r5, #0x0
    f510: f340 8092    	ble.w	0xf638 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x154> @ imm = #0x124
    f514: 00a3         	lsls	r3, r4, #0x2
    f516: 9307         	str	r3, [sp, #0x1c]
    f518: 9b1e         	ldr	r3, [sp, #0x78]
    f51a: 991d         	ldr	r1, [sp, #0x74]
    f51c: 9301         	str	r3, [sp, #0x4]
    f51e: 461a         	mov	r2, r3
    f520: eb02 0781    	add.w	r7, r2, r1, lsl #2
    f524: 2200         	movs	r2, #0x0
    f526: e9cd 2202    	strd	r2, r2, [sp, #8]
    f52a: 9b1f         	ldr	r3, [sp, #0x7c]
    f52c: 460a         	mov	r2, r1
    f52e: 3201         	adds	r2, #0x1
    f530: 3704         	adds	r7, #0x4
    f532: ea4f 0e89    	lsl.w	lr, r9, #0x2
    f536: 9209         	str	r2, [sp, #0x24]
    f538: 465a         	mov	r2, r11
    f53a: eb03 050e    	add.w	r5, r3, lr
    f53e: 46d3         	mov	r11, r10
    f540: eb06 040e    	add.w	r4, r6, lr
    f544: 469a         	mov	r10, r3
    f546: f8cd e010    	str.w	lr, [sp, #0x10]
    f54a: 4613         	mov	r3, r2
    f54c: 9906         	ldr	r1, [sp, #0x18]
    f54e: 9802         	ldr	r0, [sp, #0x8]
    f550: 9305         	str	r3, [sp, #0x14]
    f552: f07b ffb5    	bl	0x8b4c0 <__divsi3>      @ imm = #0x7bf6a
    f556: 9b1d         	ldr	r3, [sp, #0x74]
    f558: fb09 f600    	mul	r6, r9, r0
    f55c: 2b00         	cmp	r3, #0x0
    f55e: 9b05         	ldr	r3, [sp, #0x14]
    f560: db4d         	blt	0xf5fe <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x11a> @ imm = #0x9a
    f562: ee07 9a90    	vmov	s15, r9
    f566: 9a0a         	ldr	r2, [sp, #0x28]
    f568: 9903         	ldr	r1, [sp, #0xc]
    f56a: eef8 8ae7    	vcvt.f32.s32	s17, s15
    f56e: eb02 0081    	add.w	r0, r2, r1, lsl #2
    f572: 4631         	mov	r1, r6
    f574: 9a01         	ldr	r2, [sp, #0x4]
    f576: ed9f 8a3d    	vldr	s16, [pc, #244]         @ 0xf66c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x188>
    f57a: f1b9 0f00    	cmp.w	r9, #0x0
    f57e: dd0a         	ble	0xf596 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xb2> @ imm = #0x14
    f580: eb0b 0e81    	add.w	lr, r11, r1, lsl #2
    f584: 4684         	mov	r12, r0
    f586: ecbc 7a01    	vldmia	r12!, {s14}
    f58a: ecfe 7a01    	vldmia	lr!, {s15}
    f58e: ee07 8a27    	vmla.f32	s16, s14, s15
    f592: 45a4         	cmp	r12, r4
    f594: d1f7         	bne	0xf586 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xa2> @ imm = #-0x12
    f596: eef5 8a40    	vcmp.f32	s17, #0
    f59a: eef1 fa10    	vmrs	APSR_nzcv, fpscr
    f59e: d455         	bmi	0xf64c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x168> @ imm = #0xaa
    f5a0: eef1 7ae8    	vsqrt.f32	s15, s17
    f5a4: 4441         	add	r1, r8
    f5a6: ee88 7a27    	vdiv.f32	s14, s16, s15
    f5aa: eca2 7a01    	vstmia	r2!, {s14}
    f5ae: 42ba         	cmp	r2, r7
    f5b0: d1e1         	bne	0xf576 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x92> @ imm = #-0x3e
    f5b2: 9909         	ldr	r1, [sp, #0x24]
    f5b4: 9801         	ldr	r0, [sp, #0x4]
    f5b6: 9305         	str	r3, [sp, #0x14]
    f5b8: f7ff fb92    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x8dc
    f5bc: 9a04         	ldr	r2, [sp, #0x10]
    f5be: 2100         	movs	r1, #0x0
    f5c0: 4650         	mov	r0, r10
    f5c2: f7f8 eb32    	blx	0x7c28 <memset@plt>     @ imm = #-0x799c
    f5c6: 9a01         	ldr	r2, [sp, #0x4]
    f5c8: 9b05         	ldr	r3, [sp, #0x14]
    f5ca: f1b9 0f00    	cmp.w	r9, #0x0
    f5ce: f102 0004    	add.w	r0, r2, #0x4
    f5d2: dd36         	ble	0xf642 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x15e> @ imm = #0x6c
    f5d4: edd2 6a00    	vldr	s13, [r2]
    f5d8: eb03 0186    	add.w	r1, r3, r6, lsl #2
    f5dc: 4652         	mov	r2, r10
    f5de: edd2 7a00    	vldr	s15, [r2]
    f5e2: ecb1 7a01    	vldmia	r1!, {s14}
    f5e6: ee47 7a26    	vmla.f32	s15, s14, s13
    f5ea: ece2 7a01    	vstmia	r2!, {s15}
    f5ee: 42aa         	cmp	r2, r5
    f5f0: d1f5         	bne	0xf5de <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xfa> @ imm = #-0x16
    f5f2: 4287         	cmp	r7, r0
    f5f4: 4446         	add	r6, r8
    f5f6: d00d         	beq	0xf614 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x130> @ imm = #0x1a
    f5f8: ecf0 6a01    	vldmia	r0!, {s13}
    f5fc: e7ec         	b	0xf5d8 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xf4> @ imm = #-0x28
    f5fe: 9909         	ldr	r1, [sp, #0x24]
    f600: 9801         	ldr	r0, [sp, #0x4]
    f602: 9305         	str	r3, [sp, #0x14]
    f604: f7ff fb6c    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x928
    f608: 9a04         	ldr	r2, [sp, #0x10]
    f60a: 2100         	movs	r1, #0x0
    f60c: 4650         	mov	r0, r10
    f60e: f7f8 eb0c    	blx	0x7c28 <memset@plt>     @ imm = #-0x79e8
    f612: 9b05         	ldr	r3, [sp, #0x14]
    f614: 9807         	ldr	r0, [sp, #0x1c]
    f616: 9a01         	ldr	r2, [sp, #0x4]
    f618: 9904         	ldr	r1, [sp, #0x10]
    f61a: 4402         	add	r2, r0
    f61c: 9201         	str	r2, [sp, #0x4]
    f61e: 9a02         	ldr	r2, [sp, #0x8]
    f620: 448a         	add	r10, r1
    f622: 440d         	add	r5, r1
    f624: 3201         	adds	r2, #0x1
    f626: 440c         	add	r4, r1
    f628: 9908         	ldr	r1, [sp, #0x20]
    f62a: 4407         	add	r7, r0
    f62c: 9803         	ldr	r0, [sp, #0xc]
    f62e: 4291         	cmp	r1, r2
    f630: 4448         	add	r0, r9
    f632: 9202         	str	r2, [sp, #0x8]
    f634: 9003         	str	r0, [sp, #0xc]
    f636: d189         	bne	0xf54c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x68> @ imm = #-0xee
    f638: b00f         	add	sp, #0x3c
    f63a: ecbd 8b02    	vpop	{d8}
    f63e: e8bd 8ff0    	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
    f642: 42b8         	cmp	r0, r7
    f644: 4602         	mov	r2, r0
    f646: 4446         	add	r6, r8
    f648: d1bf         	bne	0xf5ca <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xe6> @ imm = #-0x82
    f64a: e7e3         	b	0xf614 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x130> @ imm = #-0x3a
    f64c: eeb0 0a68    	vmov.f32	s0, s17
    f650: e9cd 030c    	strd	r0, r3, [sp, #48]
    f654: 910b         	str	r1, [sp, #0x2c]
    f656: 9205         	str	r2, [sp, #0x14]
    f658: f7f8 ea8a    	blx	0x7b70 <sqrtf@plt>      @ imm = #-0x7aec
    f65c: eef0 7a40    	vmov.f32	s15, s0
    f660: e9dd 030c    	ldrd	r0, r3, [sp, #48]
    f664: 990b         	ldr	r1, [sp, #0x2c]
    f666: 9a05         	ldr	r2, [sp, #0x14]
    f668: e79c         	b	0xf5a4 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xc0> @ imm = #-0xc8
    f66a: bf00         	nop
    f66c: 00 00 00 00  	.word	0x00000000

