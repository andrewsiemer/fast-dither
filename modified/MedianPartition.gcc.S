
src/MedianPartition.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <Partition>:
       0:	55                   	push   %rbp
       1:	b8 20 00 00 00       	mov    $0x20,%eax
       6:	48 89 e5             	mov    %rsp,%rbp
       9:	41 57                	push   %r15
       b:	41 56                	push   %r14
       d:	41 55                	push   %r13
       f:	41 54                	push   %r12
      11:	53                   	push   %rbx
      12:	48 89 f3             	mov    %rsi,%rbx
      15:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
      19:	48 81 ec f0 00 00 00 	sub    $0xf0,%rsp
      20:	48 89 54 24 38       	mov    %rdx,0x38(%rsp)
      25:	48 89 f2             	mov    %rsi,%rdx
      28:	48 89 7c 24 58       	mov    %rdi,0x58(%rsp)
      2d:	83 e2 1f             	and    $0x1f,%edx
      30:	48 89 4c 24 48       	mov    %rcx,0x48(%rsp)
      35:	44 89 4c 24 40       	mov    %r9d,0x40(%rsp)
      3a:	48 29 d0             	sub    %rdx,%rax
      3d:	44 88 4c 24 47       	mov    %r9b,0x47(%rsp)
      42:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
      49:	00 00 
      4b:	48 83 f8 20          	cmp    $0x20,%rax
      4f:	74 0c                	je     5d <Partition+0x5d>
      51:	4c 39 c0             	cmp    %r8,%rax
      54:	49 0f 47 c0          	cmova  %r8,%rax
      58:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
      5d:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
      62:	4c 89 c0             	mov    %r8,%rax
      65:	48 29 f8             	sub    %rdi,%rax
      68:	83 e0 1f             	and    $0x1f,%eax
      6b:	48 01 f8             	add    %rdi,%rax
      6e:	49 39 c0             	cmp    %rax,%r8
      71:	0f 87 89 00 00 00    	ja     100 <Partition+0x100>
      77:	31 c0                	xor    %eax,%eax
      79:	4d 85 c0             	test   %r8,%r8
      7c:	74 67                	je     e5 <Partition+0xe5>
      7e:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
      83:	4c 8b 4c 24 48       	mov    0x48(%rsp),%r9
      88:	44 0f b6 54 24 47    	movzbl 0x47(%rsp),%r10d
      8e:	eb 40                	jmp    d0 <Partition+0xd0>
      90:	49 83 e8 01          	sub    $0x1,%r8
      94:	4a 8d 34 03          	lea    (%rbx,%r8,1),%rsi
      98:	49 39 c0             	cmp    %rax,%r8
      9b:	44 0f b6 1e          	movzbl (%rsi),%r11d
      9f:	44 88 19             	mov    %r11b,(%rcx)
      a2:	88 16                	mov    %dl,(%rsi)
      a4:	48 8d 0c 07          	lea    (%rdi,%rax,1),%rcx
      a8:	4a 8d 14 07          	lea    (%rdi,%r8,1),%rdx
      ac:	0f b6 31             	movzbl (%rcx),%esi
      af:	44 0f b6 1a          	movzbl (%rdx),%r11d
      b3:	44 88 19             	mov    %r11b,(%rcx)
      b6:	40 88 32             	mov    %sil,(%rdx)
      b9:	49 8d 0c 01          	lea    (%r9,%rax,1),%rcx
      bd:	4b 8d 14 01          	lea    (%r9,%r8,1),%rdx
      c1:	0f b6 31             	movzbl (%rcx),%esi
      c4:	44 0f b6 1a          	movzbl (%rdx),%r11d
      c8:	44 88 19             	mov    %r11b,(%rcx)
      cb:	40 88 32             	mov    %sil,(%rdx)
      ce:	76 15                	jbe    e5 <Partition+0xe5>
      d0:	48 8d 0c 03          	lea    (%rbx,%rax,1),%rcx
      d4:	0f b6 11             	movzbl (%rcx),%edx
      d7:	41 38 d2             	cmp    %dl,%r10b
      da:	72 b4                	jb     90 <Partition+0x90>
      dc:	48 83 c0 01          	add    $0x1,%rax
      e0:	49 39 c0             	cmp    %rax,%r8
      e3:	77 eb                	ja     d0 <Partition+0xd0>
      e5:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
      e9:	4c 89 c0             	mov    %r8,%rax
      ec:	5b                   	pop    %rbx
      ed:	41 5c                	pop    %r12
      ef:	41 5d                	pop    %r13
      f1:	41 5e                	pop    %r14
      f3:	41 5f                	pop    %r15
      f5:	5d                   	pop    %rbp
      f6:	c3                   	retq   
      f7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
      fe:	00 00 
     100:	4c 89 c6             	mov    %r8,%rsi
     103:	48 29 c6             	sub    %rax,%rsi
     106:	48 89 74 24 60       	mov    %rsi,0x60(%rsp)
     10b:	48 c1 6c 24 60 05    	shrq   $0x5,0x60(%rsp)
     111:	0f 31                	rdtsc  
     113:	48 89 d6             	mov    %rdx,%rsi
     116:	89 c0                	mov    %eax,%eax
     118:	4c 8b 64 24 48       	mov    0x48(%rsp),%r12
     11d:	48 c1 e6 20          	shl    $0x20,%rsi
     121:	4c 8b 7c 24 38       	mov    0x38(%rsp),%r15
     126:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
     12b:	48 09 44 24 28       	or     %rax,0x28(%rsp)
     130:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
     135:	49 01 c4             	add    %rax,%r12
     138:	49 01 c7             	add    %rax,%r15
     13b:	48 01 d8             	add    %rbx,%rax
     13e:	48 83 7c 24 60 00    	cmpq   $0x0,0x60(%rsp)
     144:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
     149:	0f 84 0a 0b 00 00    	je     c59 <Partition+0xc59>
     14f:	49 89 c2             	mov    %rax,%r10
     152:	0f 31                	rdtsc  
     154:	48 89 d6             	mov    %rdx,%rsi
     157:	0f b6 7c 24 40       	movzbl 0x40(%rsp),%edi
     15c:	89 c0                	mov    %eax,%eax
     15e:	48 c1 e6 20          	shl    $0x20,%rsi
     162:	c5 fd 6f 05 00 00 00 	vmovdqa 0x0(%rip),%ymm0        # 16a <Partition+0x16a>
     169:	00 
     16a:	48 89 74 24 20       	mov    %rsi,0x20(%rsp)
     16f:	0f b6 74 24 47       	movzbl 0x47(%rsp),%esi
     174:	48 09 44 24 20       	or     %rax,0x20(%rsp)
     179:	31 c0                	xor    %eax,%eax
     17b:	c5 fd 6f f8          	vmovdqa %ymm0,%ymm7
     17f:	89 fa                	mov    %edi,%edx
     181:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
     185:	c1 e2 18             	shl    $0x18,%edx
     188:	89 7c 24 1c          	mov    %edi,0x1c(%rsp)
     18c:	c5 fd 6f 15 00 00 00 	vmovdqa 0x0(%rip),%ymm2        # 194 <Partition+0x194>
     193:	00 
     194:	48 89 f1             	mov    %rsi,%rcx
     197:	40 88 f0             	mov    %sil,%al
     19a:	88 cc                	mov    %cl,%ah
     19c:	89 f9                	mov    %edi,%ecx
     19e:	c5 7d 6f c2          	vmovdqa %ymm2,%ymm8
     1a2:	c1 e1 10             	shl    $0x10,%ecx
     1a5:	0f b7 c0             	movzwl %ax,%eax
     1a8:	09 c8                	or     %ecx,%eax
     1aa:	09 d0                	or     %edx,%eax
     1ac:	89 44 24 78          	mov    %eax,0x78(%rsp)
     1b0:	c5 7a 10 54 24 78    	vmovss 0x78(%rsp),%xmm10
     1b6:	c5 7a 11 94 24 d0 00 	vmovss %xmm10,0xd0(%rsp)
     1bd:	00 00 
     1bf:	c4 42 7d 18 d2       	vbroadcastss %xmm10,%ymm10
     1c4:	c4 c1 7d 6f 22       	vmovdqa (%r10),%ymm4
     1c9:	c4 41 7d fc d2       	vpaddb %ymm10,%ymm0,%ymm10
     1ce:	c5 fd fc c4          	vpaddb %ymm4,%ymm0,%ymm0
     1d2:	c4 c1 7d 64 c2       	vpcmpgtb %ymm10,%ymm0,%ymm0
     1d7:	c5 7d 6f cc          	vmovdqa %ymm4,%ymm9
     1db:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
     1df:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     1e3:	0f b6 c8             	movzbl %al,%ecx
     1e6:	89 c2                	mov    %eax,%edx
     1e8:	0f b7 f8             	movzwl %ax,%edi
     1eb:	f3 0f b8 f1          	popcnt %ecx,%esi
     1ef:	c1 ea 10             	shr    $0x10,%edx
     1f2:	f3 0f b8 ff          	popcnt %edi,%edi
     1f6:	41 89 f1             	mov    %esi,%r9d
     1f9:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
     1fe:	0f b6 d2             	movzbl %dl,%edx
     201:	f3 44 0f b8 f2       	popcnt %edx,%r14d
     206:	4c 8b 1e             	mov    (%rsi),%r11
     209:	f3 0f b8 f0          	popcnt %eax,%esi
     20d:	41 89 33             	mov    %esi,(%r11)
     210:	c4 e3 f1 22 0c cd 00 	vpinsrq $0x0,0x0(,%rcx,8),%xmm1,%xmm1
     217:	00 00 00 00 
     21b:	0f b6 cc             	movzbl %ah,%ecx
     21e:	c1 e8 18             	shr    $0x18,%eax
     221:	48 83 7c 24 60 01    	cmpq   $0x1,0x60(%rsp)
     227:	c4 e3 f9 22 04 cd 00 	vpinsrq $0x0,0x0(,%rcx,8),%xmm0,%xmm0
     22e:	00 00 00 00 
     232:	c5 f1 6c e8          	vpunpcklqdq %xmm0,%xmm1,%xmm5
     236:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
     23a:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     23e:	c4 e3 f1 22 0c d5 00 	vpinsrq $0x0,0x0(,%rdx,8),%xmm1,%xmm1
     245:	00 00 00 00 
     249:	c4 e3 f9 22 04 c5 00 	vpinsrq $0x0,0x0(,%rax,8),%xmm0,%xmm0
     250:	00 00 00 00 
     254:	c5 f1 6c d8          	vpunpcklqdq %xmm0,%xmm1,%xmm3
     258:	c5 f9 28 c5          	vmovapd %xmm5,%xmm0
     25c:	c4 e3 7d 38 c3 01    	vinserti128 $0x1,%xmm3,%ymm0,%ymm0
     262:	c5 ed fc d0          	vpaddb %ymm0,%ymm2,%ymm2
     266:	c5 fd 6f c2          	vmovdqa %ymm2,%ymm0
     26a:	0f 84 cb 09 00 00    	je     c3b <Partition+0xc3b>
     270:	be 01 00 00 00       	mov    $0x1,%esi
     275:	48 89 5c 24 10       	mov    %rbx,0x10(%rsp)
     27a:	49 8d 4a 20          	lea    0x20(%r10),%rcx
     27e:	b8 04 00 00 00       	mov    $0x4,%eax
     283:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
     287:	48 89 74 24 78       	mov    %rsi,0x78(%rsp)
     28c:	44 89 cb             	mov    %r9d,%ebx
     28f:	4c 89 44 24 08       	mov    %r8,0x8(%rsp)
     294:	eb 26                	jmp    2bc <Partition+0x2bc>
     296:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     29d:	00 00 00 
     2a0:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
     2a5:	44 8b 74 24 68       	mov    0x68(%rsp),%r14d
     2aa:	c5 fd 6f e5          	vmovdqa %ymm5,%ymm4
     2ae:	8b 5c 24 70          	mov    0x70(%rsp),%ebx
     2b2:	c5 fd 6f c6          	vmovdqa %ymm6,%ymm0
     2b6:	4c 8b 1f             	mov    (%rdi),%r11
     2b9:	44 89 ef             	mov    %r13d,%edi
     2bc:	c5 fd 6f 29          	vmovdqa (%rcx),%ymm5
     2c0:	83 c3 01             	add    $0x1,%ebx
     2c3:	45 89 f6             	mov    %r14d,%r14d
     2c6:	48 c1 e3 04          	shl    $0x4,%rbx
     2ca:	49 c1 e6 04          	shl    $0x4,%r14
     2ce:	48 83 44 24 78 01    	addq   $0x1,0x78(%rsp)
     2d4:	c5 c5 fc d5          	vpaddb %ymm5,%ymm7,%ymm2
     2d8:	c4 c1 6d 64 d2       	vpcmpgtb %ymm10,%ymm2,%ymm2
     2dd:	48 83 c1 20          	add    $0x20,%rcx
     2e1:	c5 7d 6f cd          	vmovdqa %ymm5,%ymm9
     2e5:	c5 fd d7 d2          	vpmovmskb %ymm2,%edx
     2e9:	41 89 d0             	mov    %edx,%r8d
     2ec:	44 0f b6 ca          	movzbl %dl,%r9d
     2f0:	44 0f b7 ea          	movzwl %dx,%r13d
     2f4:	41 c1 e8 10          	shr    $0x10,%r8d
     2f8:	f3 41 0f b8 f1       	popcnt %r9d,%esi
     2fd:	f3 45 0f b8 ed       	popcnt %r13d,%r13d
     302:	45 0f b6 c0          	movzbl %r8b,%r8d
     306:	89 74 24 70          	mov    %esi,0x70(%rsp)
     30a:	0f b6 f6             	movzbl %dh,%esi
     30d:	f3 45 0f b8 d0       	popcnt %r8d,%r10d
     312:	44 89 54 24 68       	mov    %r10d,0x68(%rsp)
     317:	f3 44 0f b8 d2       	popcnt %edx,%r10d
     31c:	c1 ea 18             	shr    $0x18,%edx
     31f:	45 89 14 03          	mov    %r10d,(%r11,%rax,1)
     323:	c4 e3 f1 22 14 f5 00 	vpinsrq $0x0,0x0(,%rsi,8),%xmm1,%xmm2
     32a:	00 00 00 00 
     32e:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
     333:	c4 a3 f1 22 34 cd 00 	vpinsrq $0x0,0x0(,%r9,8),%xmm1,%xmm6
     33a:	00 00 00 00 
     33e:	c4 a3 f1 22 1c c5 00 	vpinsrq $0x0,0x0(,%r8,8),%xmm1,%xmm3
     345:	00 00 00 00 
     349:	c5 c9 6c f2          	vpunpcklqdq %xmm2,%xmm6,%xmm6
     34d:	c4 e3 f1 22 14 d5 00 	vpinsrq $0x0,0x0(,%rdx,8),%xmm1,%xmm2
     354:	00 00 00 00 
     358:	89 fa                	mov    %edi,%edx
     35a:	89 ff                	mov    %edi,%edi
     35c:	83 e2 0f             	and    $0xf,%edx
     35f:	48 c1 e7 05          	shl    $0x5,%rdi
     363:	48 c1 e2 05          	shl    $0x5,%rdx
     367:	c5 e1 6c d2          	vpunpcklqdq %xmm2,%xmm3,%xmm2
     36b:	c4 c1 7e 6f 9e 00 00 	vmovdqu 0x0(%r14),%ymm3
     372:	00 00 
     374:	c4 e3 4d 38 f2 01    	vinserti128 $0x1,%xmm2,%ymm6,%ymm6
     37a:	c5 fe 6f 93 00 00 00 	vmovdqu 0x0(%rbx),%ymm2
     381:	00 
     382:	c4 e3 6d 02 d3 f0    	vpblendd $0xf0,%ymm3,%ymm2,%ymm2
     388:	c4 c1 7d 6f 5c c7 e0 	vmovdqa -0x20(%r15,%rax,8),%ymm3
     38f:	c5 bd fc f6          	vpaddb %ymm6,%ymm8,%ymm6
     393:	c5 7d 6f de          	vmovdqa %ymm6,%ymm11
     397:	c4 e2 7d 00 d2       	vpshufb %ymm2,%ymm0,%ymm2
     39c:	c4 c1 7d 6f 44 c4 e0 	vmovdqa -0x20(%r12,%rax,8),%ymm0
     3a3:	c4 e2 6d 00 92 00 00 	vpshufb 0x0(%rdx),%ymm2,%ymm2
     3aa:	00 00 
     3ac:	c4 e2 5d 00 e2       	vpshufb %ymm2,%ymm4,%ymm4
     3b1:	c4 e2 65 00 da       	vpshufb %ymm2,%ymm3,%ymm3
     3b6:	c4 e2 7d 00 d2       	vpshufb %ymm2,%ymm0,%ymm2
     3bb:	c5 fd 6f 87 00 00 00 	vmovdqa 0x0(%rdi),%ymm0
     3c2:	00 
     3c3:	c4 63 5d 46 f4 01    	vperm2i128 $0x1,%ymm4,%ymm4,%ymm14
     3c9:	c4 63 65 46 eb 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm13
     3cf:	c4 63 6d 46 e2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm12
     3d5:	c5 fd db e4          	vpand  %ymm4,%ymm0,%ymm4
     3d9:	c5 fd db db          	vpand  %ymm3,%ymm0,%ymm3
     3dd:	c5 fd db d2          	vpand  %ymm2,%ymm0,%ymm2
     3e1:	c4 41 7d df f6       	vpandn %ymm14,%ymm0,%ymm14
     3e6:	c4 41 7d df ed       	vpandn %ymm13,%ymm0,%ymm13
     3eb:	c4 c1 7d df c4       	vpandn %ymm12,%ymm0,%ymm0
     3f0:	c4 c1 5d eb e6       	vpor   %ymm14,%ymm4,%ymm4
     3f5:	c4 c1 65 eb dd       	vpor   %ymm13,%ymm3,%ymm3
     3fa:	c5 ed eb c0          	vpor   %ymm0,%ymm2,%ymm0
     3fe:	c5 fd 7f 61 c0       	vmovdqa %ymm4,-0x40(%rcx)
     403:	c4 c1 7d 7f 5c c7 e0 	vmovdqa %ymm3,-0x20(%r15,%rax,8)
     40a:	c4 c1 7d 7f 44 c4 e0 	vmovdqa %ymm0,-0x20(%r12,%rax,8)
     411:	48 83 c0 04          	add    $0x4,%rax
     415:	48 39 74 24 78       	cmp    %rsi,0x78(%rsp)
     41a:	0f 85 80 fe ff ff    	jne    2a0 <Partition+0x2a0>
     420:	48 8b 5c 24 10       	mov    0x10(%rsp),%rbx
     425:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
     42a:	44 89 ef             	mov    %r13d,%edi
     42d:	8b 74 24 70          	mov    0x70(%rsp),%esi
     431:	8d 56 01             	lea    0x1(%rsi),%edx
     434:	48 8b 74 24 60       	mov    0x60(%rsp),%rsi
     439:	89 f8                	mov    %edi,%eax
     43b:	83 e0 0f             	and    $0xf,%eax
     43e:	89 f9                	mov    %edi,%ecx
     440:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
     445:	48 c1 e2 04          	shl    $0x4,%rdx
     449:	48 c1 e0 05          	shl    $0x5,%rax
     44d:	48 c1 e1 05          	shl    $0x5,%rcx
     451:	c5 fe 6f 82 00 00 00 	vmovdqu 0x0(%rdx),%ymm0
     458:	00 
     459:	8b 54 24 68          	mov    0x68(%rsp),%edx
     45d:	49 89 f5             	mov    %rsi,%r13
     460:	49 c1 e5 05          	shl    $0x5,%r13
     464:	48 c1 e2 04          	shl    $0x4,%rdx
     468:	c5 fe 6f 8a 00 00 00 	vmovdqu 0x0(%rdx),%ymm1
     46f:	00 
     470:	c4 e3 7d 02 c1 f0    	vpblendd $0xf0,%ymm1,%ymm0,%ymm0
     476:	c4 62 25 00 d8       	vpshufb %ymm0,%ymm11,%ymm11
     47b:	c5 fd 6f 81 00 00 00 	vmovdqa 0x0(%rcx),%ymm0
     482:	00 
     483:	c4 62 25 00 98 00 00 	vpshufb 0x0(%rax),%ymm11,%ymm11
     48a:	00 00 
     48c:	49 8d 45 e0          	lea    -0x20(%r13),%rax
     490:	49 8d 14 07          	lea    (%r15,%rax,1),%rdx
     494:	4c 01 e0             	add    %r12,%rax
     497:	c5 fd 6f 08          	vmovdqa (%rax),%ymm1
     49b:	c4 42 35 00 cb       	vpshufb %ymm11,%ymm9,%ymm9
     4a0:	c5 fd 6f 12          	vmovdqa (%rdx),%ymm2
     4a4:	c4 c2 75 00 cb       	vpshufb %ymm11,%ymm1,%ymm1
     4a9:	c4 c2 6d 00 d3       	vpshufb %ymm11,%ymm2,%ymm2
     4ae:	c4 c3 35 46 e1 01    	vperm2i128 $0x1,%ymm9,%ymm9,%ymm4
     4b4:	c4 e3 75 46 e9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm5
     4ba:	c4 e3 6d 46 da 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm3
     4c0:	c5 fd db c9          	vpand  %ymm1,%ymm0,%ymm1
     4c4:	c5 fd df e4          	vpandn %ymm4,%ymm0,%ymm4
     4c8:	c5 fd df db          	vpandn %ymm3,%ymm0,%ymm3
     4cc:	c4 41 7d db c9       	vpand  %ymm9,%ymm0,%ymm9
     4d1:	c5 fd db d2          	vpand  %ymm2,%ymm0,%ymm2
     4d5:	c5 fd df c5          	vpandn %ymm5,%ymm0,%ymm0
     4d9:	c5 35 eb cc          	vpor   %ymm4,%ymm9,%ymm9
     4dd:	c5 ed eb d3          	vpor   %ymm3,%ymm2,%ymm2
     4e1:	c5 f5 eb c0          	vpor   %ymm0,%ymm1,%ymm0
     4e5:	c4 21 7d 7f 4c 2f e0 	vmovdqa %ymm9,-0x20(%rdi,%r13,1)
     4ec:	c5 fd 7f 12          	vmovdqa %ymm2,(%rdx)
     4f0:	c5 fd 7f 00          	vmovdqa %ymm0,(%rax)
     4f4:	0f 31                	rdtsc  
     4f6:	48 8b 4d 10          	mov    0x10(%rbp),%rcx
     4fa:	48 c1 e2 20          	shl    $0x20,%rdx
     4fe:	89 c0                	mov    %eax,%eax
     500:	48 09 c2             	or     %rax,%rdx
     503:	48 8b 45 10          	mov    0x10(%rbp),%rax
     507:	48 8b 49 50          	mov    0x50(%rcx),%rcx
     50b:	48 2b 4c 24 20       	sub    0x20(%rsp),%rcx
     510:	4c 01 68 58          	add    %r13,0x58(%rax)
     514:	48 01 ca             	add    %rcx,%rdx
     517:	48 89 50 50          	mov    %rdx,0x50(%rax)
     51b:	0f 31                	rdtsc  
     51d:	48 89 d1             	mov    %rdx,%rcx
     520:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
     525:	89 c0                	mov    %eax,%eax
     527:	48 c1 e1 20          	shl    $0x20,%rcx
     52b:	c5 fd 6f 27          	vmovdqa (%rdi),%ymm4
     52f:	48 89 4c 24 78       	mov    %rcx,0x78(%rsp)
     534:	48 89 f1             	mov    %rsi,%rcx
     537:	48 09 44 24 78       	or     %rax,0x78(%rsp)
     53c:	4c 8b 1a             	mov    (%rdx),%r11
     53f:	48 83 e9 01          	sub    $0x1,%rcx
     543:	c4 c1 7d 6f 1f       	vmovdqa (%r15),%ymm3
     548:	c4 c1 7d 6f 14 24    	vmovdqa (%r12),%ymm2
     54e:	41 8b 03             	mov    (%r11),%eax
     551:	0f 84 fa 06 00 00    	je     c51 <Partition+0xc51>
     557:	4c 89 6c 24 70       	mov    %r13,0x70(%rsp)
     55c:	48 89 5c 24 68       	mov    %rbx,0x68(%rsp)
     561:	49 89 ca             	mov    %rcx,%r10
     564:	45 31 c9             	xor    %r9d,%r9d
     567:	45 31 f6             	xor    %r14d,%r14d
     56a:	49 89 d5             	mov    %rdx,%r13
     56d:	48 89 fb             	mov    %rdi,%rbx
     570:	eb 3a                	jmp    5ac <Partition+0x5ac>
     572:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
     578:	49 8d 49 01          	lea    0x1(%r9),%rcx
     57c:	83 fa 20             	cmp    $0x20,%edx
     57f:	b0 20                	mov    $0x20,%al
     581:	0f 46 c2             	cmovbe %edx,%eax
     584:	4c 89 ca             	mov    %r9,%rdx
     587:	49 89 c9             	mov    %rcx,%r9
     58a:	48 c1 e2 05          	shl    $0x5,%rdx
     58e:	4d 39 ca             	cmp    %r9,%r10
     591:	c5 7d 7f 04 13       	vmovdqa %ymm8,(%rbx,%rdx,1)
     596:	c4 41 7d 7f 0c 17    	vmovdqa %ymm9,(%r15,%rdx,1)
     59c:	c4 c1 7d 7f 3c 14    	vmovdqa %ymm7,(%r12,%rdx,1)
     5a2:	0f 86 1e 01 00 00    	jbe    6c6 <Partition+0x6c6>
     5a8:	4d 8b 5d 00          	mov    0x0(%r13),%r11
     5ac:	89 c7                	mov    %eax,%edi
     5ae:	48 89 ce             	mov    %rcx,%rsi
     5b1:	89 c2                	mov    %eax,%edx
     5b3:	83 e7 0f             	and    $0xf,%edi
     5b6:	48 c1 e6 05          	shl    $0x5,%rsi
     5ba:	48 c1 e2 05          	shl    $0x5,%rdx
     5be:	48 c1 e7 05          	shl    $0x5,%rdi
     5c2:	c4 c1 7d 6f 3c 34    	vmovdqa (%r12,%rsi,1),%ymm7
     5c8:	41 03 04 8b          	add    (%r11,%rcx,4),%eax
     5cc:	c5 fd 6f 87 00 00 00 	vmovdqa 0x0(%rdi),%ymm0
     5d3:	00 
     5d4:	c5 fd 6f 34 33       	vmovdqa (%rbx,%rsi,1),%ymm6
     5d9:	c4 c1 7d 6f 2c 37    	vmovdqa (%r15,%rsi,1),%ymm5
     5df:	c4 e2 4d 00 f0       	vpshufb %ymm0,%ymm6,%ymm6
     5e4:	c5 fd 6f 8a 00 00 00 	vmovdqa 0x0(%rdx),%ymm1
     5eb:	00 
     5ec:	c4 e2 55 00 e8       	vpshufb %ymm0,%ymm5,%ymm5
     5f1:	c4 e2 45 00 c0       	vpshufb %ymm0,%ymm7,%ymm0
     5f6:	c4 63 4d 46 c6 01    	vperm2i128 $0x1,%ymm6,%ymm6,%ymm8
     5fc:	c4 e3 55 46 fd 01    	vperm2i128 $0x1,%ymm5,%ymm5,%ymm7
     602:	c4 63 7d 46 c8 01    	vperm2i128 $0x1,%ymm0,%ymm0,%ymm9
     608:	c4 41 75 df c0       	vpandn %ymm8,%ymm1,%ymm8
     60d:	c5 f5 db c0          	vpand  %ymm0,%ymm1,%ymm0
     611:	c5 f5 df ff          	vpandn %ymm7,%ymm1,%ymm7
     615:	c5 f5 db f6          	vpand  %ymm6,%ymm1,%ymm6
     619:	c5 f5 db ed          	vpand  %ymm5,%ymm1,%ymm5
     61d:	c4 c1 75 df c9       	vpandn %ymm9,%ymm1,%ymm1
     622:	c4 c1 4d eb f0       	vpor   %ymm8,%ymm6,%ymm6
     627:	c5 d5 eb ef          	vpor   %ymm7,%ymm5,%ymm5
     62b:	c5 fd eb c9          	vpor   %ymm1,%ymm0,%ymm1
     62f:	c5 fd 6f 82 00 00 00 	vmovdqa 0x0(%rdx),%ymm0
     636:	00 
     637:	89 c2                	mov    %eax,%edx
     639:	83 e8 20             	sub    $0x20,%eax
     63c:	c5 7d df e6          	vpandn %ymm6,%ymm0,%ymm12
     640:	41 0f 48 c6          	cmovs  %r14d,%eax
     644:	c5 7d db c4          	vpand  %ymm4,%ymm0,%ymm8
     648:	85 c0                	test   %eax,%eax
     64a:	c5 7d df dd          	vpandn %ymm5,%ymm0,%ymm11
     64e:	c5 7d db cb          	vpand  %ymm3,%ymm0,%ymm9
     652:	c5 7d df d1          	vpandn %ymm1,%ymm0,%ymm10
     656:	c5 fd db fa          	vpand  %ymm2,%ymm0,%ymm7
     65a:	c5 fd db f6          	vpand  %ymm6,%ymm0,%ymm6
     65e:	c5 fd df e4          	vpandn %ymm4,%ymm0,%ymm4
     662:	c5 fd db ed          	vpand  %ymm5,%ymm0,%ymm5
     666:	c5 fd df db          	vpandn %ymm3,%ymm0,%ymm3
     66a:	c5 fd db c9          	vpand  %ymm1,%ymm0,%ymm1
     66e:	c5 fd df d2          	vpandn %ymm2,%ymm0,%ymm2
     672:	c4 41 1d eb c0       	vpor   %ymm8,%ymm12,%ymm8
     677:	c4 41 25 eb c9       	vpor   %ymm9,%ymm11,%ymm9
     67c:	c5 ad eb ff          	vpor   %ymm7,%ymm10,%ymm7
     680:	c5 dd eb e6          	vpor   %ymm6,%ymm4,%ymm4
     684:	c5 e5 eb dd          	vpor   %ymm5,%ymm3,%ymm3
     688:	c5 ed eb d1          	vpor   %ymm1,%ymm2,%ymm2
     68c:	0f 84 e6 fe ff ff    	je     578 <Partition+0x578>
     692:	49 8d 4a ff          	lea    -0x1(%r10),%rcx
     696:	4c 89 d2             	mov    %r10,%rdx
     699:	48 c1 e2 05          	shl    $0x5,%rdx
     69d:	49 89 ca             	mov    %rcx,%r10
     6a0:	c5 fd 7f 24 13       	vmovdqa %ymm4,(%rbx,%rdx,1)
     6a5:	c5 7d 7f c4          	vmovdqa %ymm8,%ymm4
     6a9:	4d 39 ca             	cmp    %r9,%r10
     6ac:	c4 c1 7d 7f 1c 17    	vmovdqa %ymm3,(%r15,%rdx,1)
     6b2:	c5 7d 7f cb          	vmovdqa %ymm9,%ymm3
     6b6:	c4 c1 7d 7f 14 14    	vmovdqa %ymm2,(%r12,%rdx,1)
     6bc:	c5 fd 6f d7          	vmovdqa %ymm7,%ymm2
     6c0:	0f 87 e2 fe ff ff    	ja     5a8 <Partition+0x5a8>
     6c6:	4c 8b 6c 24 70       	mov    0x70(%rsp),%r13
     6cb:	48 8b 5c 24 68       	mov    0x68(%rsp),%rbx
     6d0:	49 c1 e1 05          	shl    $0x5,%r9
     6d4:	c5 fd 6f 3d 00 00 00 	vmovdqa 0x0(%rip),%ymm7        # 6dc <Partition+0x6dc>
     6db:	00 
     6dc:	c5 7d 6f 05 00 00 00 	vmovdqa 0x0(%rip),%ymm8        # 6e4 <Partition+0x6e4>
     6e3:	00 
     6e4:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
     6e9:	c4 a1 7d 7f 24 08    	vmovdqa %ymm4,(%rax,%r9,1)
     6ef:	c4 81 7d 7f 1c 0f    	vmovdqa %ymm3,(%r15,%r9,1)
     6f5:	c4 81 7d 7f 14 0c    	vmovdqa %ymm2,(%r12,%r9,1)
     6fb:	0f 31                	rdtsc  
     6fd:	48 8b 7d 10          	mov    0x10(%rbp),%rdi
     701:	48 c1 e2 20          	shl    $0x20,%rdx
     705:	89 c0                	mov    %eax,%eax
     707:	48 09 c2             	or     %rax,%rdx
     70a:	48 8b 4f 60          	mov    0x60(%rdi),%rcx
     70e:	48 2b 4c 24 78       	sub    0x78(%rsp),%rcx
     713:	4c 01 6f 68          	add    %r13,0x68(%rdi)
     717:	48 01 ca             	add    %rcx,%rdx
     71a:	48 89 57 60          	mov    %rdx,0x60(%rdi)
     71e:	0f 31                	rdtsc  
     720:	48 8b 4f 40          	mov    0x40(%rdi),%rcx
     724:	4c 03 4c 24 50       	add    0x50(%rsp),%r9
     729:	48 c1 e2 20          	shl    $0x20,%rdx
     72d:	48 2b 4c 24 28       	sub    0x28(%rsp),%rcx
     732:	89 c0                	mov    %eax,%eax
     734:	4c 01 6f 48          	add    %r13,0x48(%rdi)
     738:	48 09 c2             	or     %rax,%rdx
     73b:	48 01 ca             	add    %rcx,%rdx
     73e:	4d 39 c8             	cmp    %r9,%r8
     741:	48 89 57 40          	mov    %rdx,0x40(%rdi)
     745:	76 28                	jbe    76f <Partition+0x76f>
     747:	0f b6 44 24 40       	movzbl 0x40(%rsp),%eax
     74c:	42 3a 04 0b          	cmp    (%rbx,%r9,1),%al
     750:	72 1d                	jb     76f <Partition+0x76f>
     752:	0f b6 44 24 47       	movzbl 0x47(%rsp),%eax
     757:	eb 0d                	jmp    766 <Partition+0x766>
     759:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
     760:	42 3a 04 0b          	cmp    (%rbx,%r9,1),%al
     764:	72 09                	jb     76f <Partition+0x76f>
     766:	49 83 c1 01          	add    $0x1,%r9
     76a:	4d 39 c8             	cmp    %r9,%r8
     76d:	77 f1                	ja     760 <Partition+0x760>
     76f:	49 8d 41 e0          	lea    -0x20(%r9),%rax
     773:	c5 fe 6f 03          	vmovdqu (%rbx),%ymm0
     777:	31 d2                	xor    %edx,%edx
     779:	49 83 f9 20          	cmp    $0x20,%r9
     77d:	4c 8b 64 24 48       	mov    0x48(%rsp),%r12
     782:	0f b6 7c 24 47       	movzbl 0x47(%rsp),%edi
     787:	48 0f 4d d0          	cmovge %rax,%rdx
     78b:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
     790:	c5 fd 7f 84 24 90 00 	vmovdqa %ymm0,0x90(%rsp)
     797:	00 00 
     799:	48 8d 34 13          	lea    (%rbx,%rdx,1),%rsi
     79d:	44 8b 6c 24 1c       	mov    0x1c(%rsp),%r13d
     7a2:	c5 fe 6f 9c 24 90 00 	vmovdqu 0x90(%rsp),%ymm3
     7a9:	00 00 
     7ab:	c5 fe 6f 00          	vmovdqu (%rax),%ymm0
     7af:	48 8d 0c 10          	lea    (%rax,%rdx,1),%rcx
     7b3:	4c 01 e2             	add    %r12,%rdx
     7b6:	31 c0                	xor    %eax,%eax
     7b8:	45 89 ef             	mov    %r13d,%r15d
     7bb:	41 c1 e5 18          	shl    $0x18,%r13d
     7bf:	c5 fd 7f 84 24 b0 00 	vmovdqa %ymm0,0xb0(%rsp)
     7c6:	00 00 
     7c8:	40 88 f8             	mov    %dil,%al
     7cb:	41 c1 e7 10          	shl    $0x10,%r15d
     7cf:	c5 e5 fc e7          	vpaddb %ymm7,%ymm3,%ymm4
     7d3:	c4 c1 7e 6f 04 24    	vmovdqu (%r12),%ymm0
     7d9:	c5 fe 6f 94 24 b0 00 	vmovdqu 0xb0(%rsp),%ymm2
     7e0:	00 00 
     7e2:	c5 fd 7f 84 24 d0 00 	vmovdqa %ymm0,0xd0(%rsp)
     7e9:	00 00 
     7eb:	c5 7e 6f 0e          	vmovdqu (%rsi),%ymm9
     7ef:	c5 fe 6f 31          	vmovdqu (%rcx),%ymm6
     7f3:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
     7f8:	c5 fe 6f 2a          	vmovdqu (%rdx),%ymm5
     7fc:	48 89 fa             	mov    %rdi,%rdx
     7ff:	88 d4                	mov    %dl,%ah
     801:	c5 fe 6f 8c 24 d0 00 	vmovdqu 0xd0(%rsp),%ymm1
     808:	00 00 
     80a:	0f b7 c0             	movzwl %ax,%eax
     80d:	44 09 f8             	or     %r15d,%eax
     810:	44 09 e8             	or     %r13d,%eax
     813:	89 44 24 70          	mov    %eax,0x70(%rsp)
     817:	c5 fa 10 44 24 70    	vmovss 0x70(%rsp),%xmm0
     81d:	c4 e2 7d 18 c0       	vbroadcastss %xmm0,%ymm0
     822:	c5 c5 fc c0          	vpaddb %ymm0,%ymm7,%ymm0
     826:	c5 dd 64 c0          	vpcmpgtb %ymm0,%ymm4,%ymm0
     82a:	c5 d9 ef e4          	vpxor  %xmm4,%xmm4,%xmm4
     82e:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
     832:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     836:	44 0f b6 d8          	movzbl %al,%r11d
     83a:	0f b6 d4             	movzbl %ah,%edx
     83d:	41 89 c2             	mov    %eax,%r10d
     840:	45 89 dc             	mov    %r11d,%r12d
     843:	c4 e3 f9 22 04 d5 00 	vpinsrq $0x0,0x0(,%rdx,8),%xmm0,%xmm0
     84a:	00 00 00 00 
     84e:	41 c1 ea 10          	shr    $0x10,%r10d
     852:	f3 44 0f b8 f0       	popcnt %eax,%r14d
     857:	0f b7 f8             	movzwl %ax,%edi
     85a:	c4 a3 d9 22 24 e5 00 	vpinsrq $0x0,0x0(,%r12,8),%xmm4,%xmm4
     861:	00 00 00 00 
     865:	45 0f b6 d2          	movzbl %r10b,%r10d
     869:	45 89 d4             	mov    %r10d,%r12d
     86c:	c1 e8 18             	shr    $0x18,%eax
     86f:	f3 45 0f b8 db       	popcnt %r11d,%r11d
     874:	f3 45 0f b8 d2       	popcnt %r10d,%r10d
     879:	41 83 c3 01          	add    $0x1,%r11d
     87d:	f3 0f b8 ff          	popcnt %edi,%edi
     881:	45 89 d2             	mov    %r10d,%r10d
     884:	49 c1 e3 04          	shl    $0x4,%r11
     888:	89 fa                	mov    %edi,%edx
     88a:	49 c1 e2 04          	shl    $0x4,%r10
     88e:	83 e2 0f             	and    $0xf,%edx
     891:	89 ff                	mov    %edi,%edi
     893:	c5 59 6c d8          	vpunpcklqdq %xmm0,%xmm4,%xmm11
     897:	48 c1 e7 05          	shl    $0x5,%rdi
     89b:	c5 d9 ef e4          	vpxor  %xmm4,%xmm4,%xmm4
     89f:	4d 63 f6             	movslq %r14d,%r14
     8a2:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
     8a6:	4d 29 f1             	sub    %r14,%r9
     8a9:	c4 a3 d9 22 24 e5 00 	vpinsrq $0x0,0x0(,%r12,8),%xmm4,%xmm4
     8b0:	00 00 00 00 
     8b4:	c4 e3 f9 22 04 c5 00 	vpinsrq $0x0,0x0(,%rax,8),%xmm0,%xmm0
     8bb:	00 00 00 00 
     8bf:	48 89 d0             	mov    %rdx,%rax
     8c2:	48 c1 e0 05          	shl    $0x5,%rax
     8c6:	c5 59 6c d0          	vpunpcklqdq %xmm0,%xmm4,%xmm10
     8ca:	c4 c1 7e 6f 83 00 00 	vmovdqu 0x0(%r11),%ymm0
     8d1:	00 00 
     8d3:	c5 79 29 dc          	vmovapd %xmm11,%xmm4
     8d7:	c4 c3 5d 38 e2 01    	vinserti128 $0x1,%xmm10,%ymm4,%ymm4
     8dd:	c4 41 7e 6f 92 00 00 	vmovdqu 0x0(%r10),%ymm10
     8e4:	00 00 
     8e6:	c4 c3 7d 02 c2 f0    	vpblendd $0xf0,%ymm10,%ymm0,%ymm0
     8ec:	c5 bd fc e4          	vpaddb %ymm4,%ymm8,%ymm4
     8f0:	c4 e2 5d 00 c0       	vpshufb %ymm0,%ymm4,%ymm0
     8f5:	c4 e2 7d 00 80 00 00 	vpshufb 0x0(%rax),%ymm0,%ymm0
     8fc:	00 00 
     8fe:	b8 20 00 00 00       	mov    $0x20,%eax
     903:	4c 29 f0             	sub    %r14,%rax
     906:	49 39 c1             	cmp    %rax,%r9
     909:	c4 e2 65 00 d8       	vpshufb %ymm0,%ymm3,%ymm3
     90e:	4c 0f 4c c8          	cmovl  %rax,%r9
     912:	c4 e2 6d 00 d0       	vpshufb %ymm0,%ymm2,%ymm2
     917:	c4 e2 75 00 c8       	vpshufb %ymm0,%ymm1,%ymm1
     91c:	c5 fd 6f 87 00 00 00 	vmovdqa 0x0(%rdi),%ymm0
     923:	00 
     924:	c4 63 65 46 d3 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm10
     92a:	c4 e3 6d 46 e2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm4
     930:	c4 63 75 46 d9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm11
     936:	c4 41 7d df d2       	vpandn %ymm10,%ymm0,%ymm10
     93b:	c5 fd db c9          	vpand  %ymm1,%ymm0,%ymm1
     93f:	c5 fd df e4          	vpandn %ymm4,%ymm0,%ymm4
     943:	c5 fd db db          	vpand  %ymm3,%ymm0,%ymm3
     947:	c5 fd db d2          	vpand  %ymm2,%ymm0,%ymm2
     94b:	c4 c1 7d df c3       	vpandn %ymm11,%ymm0,%ymm0
     950:	c4 c1 65 eb da       	vpor   %ymm10,%ymm3,%ymm3
     955:	c5 ed eb d4          	vpor   %ymm4,%ymm2,%ymm2
     959:	c5 f5 eb c0          	vpor   %ymm0,%ymm1,%ymm0
     95d:	c5 fe 7f 9c 24 90 00 	vmovdqu %ymm3,0x90(%rsp)
     964:	00 00 
     966:	c5 fe 7f 94 24 b0 00 	vmovdqu %ymm2,0xb0(%rsp)
     96d:	00 00 
     96f:	c5 fe 7f 84 24 d0 00 	vmovdqu %ymm0,0xd0(%rsp)
     976:	00 00 
     978:	c5 fd 6f 05 00 00 00 	vmovdqa 0x0(%rip),%ymm0        # 980 <Partition+0x980>
     97f:	00 
     980:	c4 e2 35 00 d8       	vpshufb %ymm0,%ymm9,%ymm3
     985:	c4 e2 4d 00 d0       	vpshufb %ymm0,%ymm6,%ymm2
     98a:	c4 e2 55 00 c8       	vpshufb %ymm0,%ymm5,%ymm1
     98f:	c4 e3 65 46 db 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm3
     995:	c4 e3 6d 46 d2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm2
     99b:	c4 e3 75 46 c9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm1
     9a1:	c5 fe 7f 1b          	vmovdqu %ymm3,(%rbx)
     9a5:	4c 8b 74 24 38       	mov    0x38(%rsp),%r14
     9aa:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
     9ae:	c4 c1 7e 7f 16       	vmovdqu %ymm2,(%r14)
     9b3:	4c 8b 64 24 48       	mov    0x48(%rsp),%r12
     9b8:	c4 c1 7e 7f 0c 24    	vmovdqu %ymm1,(%r12)
     9be:	4c 89 e7             	mov    %r12,%rdi
     9c1:	c5 fd 6f 8c 24 90 00 	vmovdqa 0x90(%rsp),%ymm1
     9c8:	00 00 
     9ca:	c5 fe 7f 0e          	vmovdqu %ymm1,(%rsi)
     9ce:	c5 fd 6f 8c 24 b0 00 	vmovdqa 0xb0(%rsp),%ymm1
     9d5:	00 00 
     9d7:	c5 fe 7f 09          	vmovdqu %ymm1,(%rcx)
     9db:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
     9e0:	c5 fd 6f 8c 24 d0 00 	vmovdqa 0xd0(%rsp),%ymm1
     9e7:	00 00 
     9e9:	c5 fe 7f 08          	vmovdqu %ymm1,(%rax)
     9ed:	49 8d 40 e0          	lea    -0x20(%r8),%rax
     9f1:	48 8d 34 03          	lea    (%rbx,%rax,1),%rsi
     9f5:	49 39 c1             	cmp    %rax,%r9
     9f8:	48 89 c2             	mov    %rax,%rdx
     9fb:	49 0f 46 d1          	cmovbe %r9,%rdx
     9ff:	49 8d 0c 06          	lea    (%r14,%rax,1),%rcx
     a03:	48 01 c7             	add    %rax,%rdi
     a06:	c5 fe 6f 0e          	vmovdqu (%rsi),%ymm1
     a0a:	4c 8d 14 13          	lea    (%rbx,%rdx,1),%r10
     a0e:	0f b6 5c 24 47       	movzbl 0x47(%rsp),%ebx
     a13:	31 c0                	xor    %eax,%eax
     a15:	4d 8d 1c 16          	lea    (%r14,%rdx,1),%r11
     a19:	4c 01 e2             	add    %r12,%rdx
     a1c:	c5 fd 7f 8c 24 90 00 	vmovdqa %ymm1,0x90(%rsp)
     a23:	00 00 
     a25:	c5 fe 6f 09          	vmovdqu (%rcx),%ymm1
     a29:	88 d8                	mov    %bl,%al
     a2b:	88 dc                	mov    %bl,%ah
     a2d:	c5 fe 6f a4 24 90 00 	vmovdqu 0x90(%rsp),%ymm4
     a34:	00 00 
     a36:	c5 fd 7f 8c 24 b0 00 	vmovdqa %ymm1,0xb0(%rsp)
     a3d:	00 00 
     a3f:	0f b7 c0             	movzwl %ax,%eax
     a42:	44 09 f8             	or     %r15d,%eax
     a45:	c5 fe 6f 0f          	vmovdqu (%rdi),%ymm1
     a49:	44 09 e8             	or     %r13d,%eax
     a4c:	c5 fe 6f 9c 24 b0 00 	vmovdqu 0xb0(%rsp),%ymm3
     a53:	00 00 
     a55:	c5 fd 7f 8c 24 d0 00 	vmovdqa %ymm1,0xd0(%rsp)
     a5c:	00 00 
     a5e:	c4 41 7e 6f 12       	vmovdqu (%r10),%ymm10
     a63:	c4 41 7e 6f 0b       	vmovdqu (%r11),%ymm9
     a68:	c5 fe 6f 32          	vmovdqu (%rdx),%ymm6
     a6c:	89 44 24 78          	mov    %eax,0x78(%rsp)
     a70:	c5 fa 10 4c 24 78    	vmovss 0x78(%rsp),%xmm1
     a76:	c4 e2 7d 18 c9       	vbroadcastss %xmm1,%ymm1
     a7b:	c5 fe 6f 94 24 d0 00 	vmovdqu 0xd0(%rsp),%ymm2
     a82:	00 00 
     a84:	c5 c5 fc c9          	vpaddb %ymm1,%ymm7,%ymm1
     a88:	c5 dd fc ff          	vpaddb %ymm7,%ymm4,%ymm7
     a8c:	c5 c5 64 f9          	vpcmpgtb %ymm1,%ymm7,%ymm7
     a90:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
     a94:	c5 fd d7 c7          	vpmovmskb %ymm7,%eax
     a98:	44 0f b6 e8          	movzbl %al,%r13d
     a9c:	41 89 c4             	mov    %eax,%r12d
     a9f:	f3 44 0f b8 f0       	popcnt %eax,%r14d
     aa4:	44 89 eb             	mov    %r13d,%ebx
     aa7:	41 c1 ec 10          	shr    $0x10,%r12d
     aab:	44 0f b7 f8          	movzwl %ax,%r15d
     aaf:	c4 e3 d1 22 2c dd 00 	vpinsrq $0x0,0x0(,%rbx,8),%xmm5,%xmm5
     ab6:	00 00 00 00 
     aba:	0f b6 dc             	movzbl %ah,%ebx
     abd:	45 0f b6 e4          	movzbl %r12b,%r12d
     ac1:	c1 e8 18             	shr    $0x18,%eax
     ac4:	f3 45 0f b8 ed       	popcnt %r13d,%r13d
     ac9:	c4 e3 f1 22 0c dd 00 	vpinsrq $0x0,0x0(,%rbx,8),%xmm1,%xmm1
     ad0:	00 00 00 00 
     ad4:	44 89 e3             	mov    %r12d,%ebx
     ad7:	f3 45 0f b8 e4       	popcnt %r12d,%r12d
     adc:	41 83 c5 01          	add    $0x1,%r13d
     ae0:	f3 45 0f b8 ff       	popcnt %r15d,%r15d
     ae5:	45 89 e4             	mov    %r12d,%r12d
     ae8:	49 c1 e5 04          	shl    $0x4,%r13
     aec:	49 c1 e4 04          	shl    $0x4,%r12
     af0:	c5 51 6c d9          	vpunpcklqdq %xmm1,%xmm5,%xmm11
     af4:	c5 d1 ef ed          	vpxor  %xmm5,%xmm5,%xmm5
     af8:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
     afc:	c4 e3 d1 22 2c dd 00 	vpinsrq $0x0,0x0(,%rbx,8),%xmm5,%xmm5
     b03:	00 00 00 00 
     b07:	44 89 fb             	mov    %r15d,%ebx
     b0a:	83 e3 0f             	and    $0xf,%ebx
     b0d:	c4 e3 f1 22 0c c5 00 	vpinsrq $0x0,0x0(,%rax,8),%xmm1,%xmm1
     b14:	00 00 00 00 
     b18:	48 c1 e3 05          	shl    $0x5,%rbx
     b1c:	44 89 f8             	mov    %r15d,%eax
     b1f:	48 c1 e0 05          	shl    $0x5,%rax
     b23:	c5 d1 6c f9          	vpunpcklqdq %xmm1,%xmm5,%xmm7
     b27:	c4 c1 7e 6f ac 24 00 	vmovdqu 0x0(%r12),%ymm5
     b2e:	00 00 00 
     b31:	c5 79 29 d9          	vmovapd %xmm11,%xmm1
     b35:	c4 e3 75 38 cf 01    	vinserti128 $0x1,%xmm7,%ymm1,%ymm1
     b3b:	c5 3d fc c1          	vpaddb %ymm1,%ymm8,%ymm8
     b3f:	c4 c1 7e 6f 8d 00 00 	vmovdqu 0x0(%r13),%ymm1
     b46:	00 00 
     b48:	c4 e3 75 02 cd f0    	vpblendd $0xf0,%ymm5,%ymm1,%ymm1
     b4e:	c4 e2 3d 00 c9       	vpshufb %ymm1,%ymm8,%ymm1
     b53:	c4 e2 75 00 8b 00 00 	vpshufb 0x0(%rbx),%ymm1,%ymm1
     b5a:	00 00 
     b5c:	c4 e2 5d 00 e1       	vpshufb %ymm1,%ymm4,%ymm4
     b61:	c4 e2 65 00 d9       	vpshufb %ymm1,%ymm3,%ymm3
     b66:	c4 e2 6d 00 d1       	vpshufb %ymm1,%ymm2,%ymm2
     b6b:	c5 fd 6f 88 00 00 00 	vmovdqa 0x0(%rax),%ymm1
     b72:	00 
     b73:	49 63 c6             	movslq %r14d,%rax
     b76:	c4 e3 5d 46 fc 01    	vperm2i128 $0x1,%ymm4,%ymm4,%ymm7
     b7c:	49 29 c0             	sub    %rax,%r8
     b7f:	c4 e3 65 46 eb 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm5
     b85:	b8 20 00 00 00       	mov    $0x20,%eax
     b8a:	c4 63 6d 46 c2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm8
     b90:	44 29 f0             	sub    %r14d,%eax
     b93:	c5 f5 db d2          	vpand  %ymm2,%ymm1,%ymm2
     b97:	49 01 c1             	add    %rax,%r9
     b9a:	c5 f5 df ff          	vpandn %ymm7,%ymm1,%ymm7
     b9e:	4d 39 c1             	cmp    %r8,%r9
     ba1:	c5 f5 df ed          	vpandn %ymm5,%ymm1,%ymm5
     ba5:	4d 0f 46 c1          	cmovbe %r9,%r8
     ba9:	c5 f5 db e4          	vpand  %ymm4,%ymm1,%ymm4
     bad:	c5 f5 db db          	vpand  %ymm3,%ymm1,%ymm3
     bb1:	c4 c1 75 df c8       	vpandn %ymm8,%ymm1,%ymm1
     bb6:	c5 dd eb e7          	vpor   %ymm7,%ymm4,%ymm4
     bba:	c5 e5 eb dd          	vpor   %ymm5,%ymm3,%ymm3
     bbe:	c5 ed eb c9          	vpor   %ymm1,%ymm2,%ymm1
     bc2:	c5 fe 7f a4 24 90 00 	vmovdqu %ymm4,0x90(%rsp)
     bc9:	00 00 
     bcb:	c4 e2 2d 00 d0       	vpshufb %ymm0,%ymm10,%ymm2
     bd0:	c5 fe 7f 9c 24 b0 00 	vmovdqu %ymm3,0xb0(%rsp)
     bd7:	00 00 
     bd9:	c5 fe 7f 8c 24 d0 00 	vmovdqu %ymm1,0xd0(%rsp)
     be0:	00 00 
     be2:	c4 e2 35 00 c8       	vpshufb %ymm0,%ymm9,%ymm1
     be7:	c4 e2 4d 00 c0       	vpshufb %ymm0,%ymm6,%ymm0
     bec:	c4 e3 6d 46 d2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm2
     bf2:	c4 e3 75 46 c9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm1
     bf8:	c4 e3 7d 46 c0 01    	vperm2i128 $0x1,%ymm0,%ymm0,%ymm0
     bfe:	c5 fe 7f 16          	vmovdqu %ymm2,(%rsi)
     c02:	c5 fe 7f 09          	vmovdqu %ymm1,(%rcx)
     c06:	c5 fe 7f 07          	vmovdqu %ymm0,(%rdi)
     c0a:	c5 fd 6f 84 24 90 00 	vmovdqa 0x90(%rsp),%ymm0
     c11:	00 00 
     c13:	c4 c1 7e 7f 02       	vmovdqu %ymm0,(%r10)
     c18:	c5 fd 6f 84 24 b0 00 	vmovdqa 0xb0(%rsp),%ymm0
     c1f:	00 00 
     c21:	c4 c1 7e 7f 03       	vmovdqu %ymm0,(%r11)
     c26:	c5 fd 6f 84 24 d0 00 	vmovdqa 0xd0(%rsp),%ymm0
     c2d:	00 00 
     c2f:	c5 fe 7f 02          	vmovdqu %ymm0,(%rdx)
     c33:	c5 f8 77             	vzeroupper 
     c36:	e9 aa f4 ff ff       	jmpq   e5 <Partition+0xe5>
     c3b:	c5 7d 6f da          	vmovdqa %ymm2,%ymm11
     c3f:	44 89 74 24 68       	mov    %r14d,0x68(%rsp)
     c44:	44 89 4c 24 70       	mov    %r9d,0x70(%rsp)
     c49:	44 89 ce             	mov    %r9d,%esi
     c4c:	e9 e0 f7 ff ff       	jmpq   431 <Partition+0x431>
     c51:	45 31 c9             	xor    %r9d,%r9d
     c54:	e9 8b fa ff ff       	jmpq   6e4 <Partition+0x6e4>
     c59:	b9 00 00 00 00       	mov    $0x0,%ecx
     c5e:	ba 00 02 00 00       	mov    $0x200,%edx
     c63:	be 00 00 00 00       	mov    $0x0,%esi
     c68:	bf 00 00 00 00       	mov    $0x0,%edi
     c6d:	e8 00 00 00 00       	callq  c72 <Partition+0xc72>
     c72:	0f 1f 40 00          	nopl   0x0(%rax)
     c76:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     c7d:	00 00 00 

0000000000000c80 <MedianPartition>:
     c80:	41 57                	push   %r15
     c82:	4c 89 c0             	mov    %r8,%rax
     c85:	48 d1 e8             	shr    %rax
     c88:	41 56                	push   %r14
     c8a:	41 55                	push   %r13
     c8c:	41 54                	push   %r12
     c8e:	55                   	push   %rbp
     c8f:	53                   	push   %rbx
     c90:	48 83 ec 68          	sub    $0x68,%rsp
     c94:	48 85 f6             	test   %rsi,%rsi
     c97:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
     c9c:	48 89 74 24 58       	mov    %rsi,0x58(%rsp)
     ca1:	48 89 54 24 48       	mov    %rdx,0x48(%rsp)
     ca6:	48 89 4c 24 50       	mov    %rcx,0x50(%rsp)
     cab:	4c 89 44 24 40       	mov    %r8,0x40(%rsp)
     cb0:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
     cb5:	0f 84 fc 02 00 00    	je     fb7 <MedianPartition+0x337>
     cbb:	48 83 7c 24 48 00    	cmpq   $0x0,0x48(%rsp)
     cc1:	0f 84 d7 02 00 00    	je     f9e <MedianPartition+0x31e>
     cc7:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
     ccd:	0f 84 80 02 00 00    	je     f53 <MedianPartition+0x2d3>
     cd3:	48 83 7c 24 40 00    	cmpq   $0x0,0x40(%rsp)
     cd9:	0f 84 5b 02 00 00    	je     f3a <MedianPartition+0x2ba>
     cdf:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
     ce4:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
     ce9:	48 39 c3             	cmp    %rax,%rbx
     cec:	0f 86 f7 02 00 00    	jbe    fe9 <MedianPartition+0x369>
     cf2:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
     cf7:	48 8b 44 24 50       	mov    0x50(%rsp),%rax
     cfc:	4c 89 cd             	mov    %r9,%rbp
     cff:	4c 8b 6c 24 58       	mov    0x58(%rsp),%r13
     d04:	41 bc ff ff ff ff    	mov    $0xffffffff,%r12d
     d0a:	c6 44 24 17 00       	movb   $0x0,0x17(%rsp)
     d0f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
     d14:	48 8b 44 24 48       	mov    0x48(%rsp),%rax
     d19:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
     d1e:	66 90                	xchg   %ax,%ax
     d20:	44 38 64 24 17       	cmp    %r12b,0x17(%rsp)
     d25:	0f 87 c5 00 00 00    	ja     df0 <MedianPartition+0x170>
     d2b:	48 83 fb 01          	cmp    $0x1,%rbx
     d2f:	0f 86 bb 00 00 00    	jbe    df0 <MedianPartition+0x170>
     d35:	e8 00 00 00 00       	callq  d3a <MedianPartition+0xba>
     d3a:	31 d2                	xor    %edx,%edx
     d3c:	89 c0                	mov    %eax,%eax
     d3e:	48 f7 f3             	div    %rbx
     d41:	45 0f b6 5c 15 00    	movzbl 0x0(%r13,%rdx,1),%r11d
     d47:	44 3a 5c 24 17       	cmp    0x17(%rsp),%r11b
     d4c:	0f 82 b6 01 00 00    	jb     f08 <MedianPartition+0x288>
     d52:	45 38 e3             	cmp    %r12b,%r11b
     d55:	45 89 df             	mov    %r11d,%r15d
     d58:	45 0f 47 fc          	cmova  %r12d,%r15d
     d5c:	0f 31                	rdtsc  
     d5e:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
     d63:	49 89 d6             	mov    %rdx,%r14
     d66:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
     d6b:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
     d70:	49 c1 e6 20          	shl    $0x20,%r14
     d74:	89 c0                	mov    %eax,%eax
     d76:	48 89 2c 24          	mov    %rbp,(%rsp)
     d7a:	45 0f b6 cf          	movzbl %r15b,%r9d
     d7e:	49 89 d8             	mov    %rbx,%r8
     d81:	4c 89 ee             	mov    %r13,%rsi
     d84:	49 09 c6             	or     %rax,%r14
     d87:	e8 74 f2 ff ff       	callq  0 <Partition>
     d8c:	48 89 c1             	mov    %rax,%rcx
     d8f:	0f 31                	rdtsc  
     d91:	48 c1 e2 20          	shl    $0x20,%rdx
     d95:	89 c0                	mov    %eax,%eax
     d97:	48 09 c2             	or     %rax,%rdx
     d9a:	48 39 cb             	cmp    %rcx,%rbx
     d9d:	0f 82 4c 01 00 00    	jb     eef <MedianPartition+0x26f>
     da3:	4c 29 f2             	sub    %r14,%rdx
     da6:	48 01 5d 18          	add    %rbx,0x18(%rbp)
     daa:	48 01 55 10          	add    %rdx,0x10(%rbp)
     dae:	48 3b 4c 24 28       	cmp    0x28(%rsp),%rcx
     db3:	0f 87 c7 00 00 00    	ja     e80 <MedianPartition+0x200>
     db9:	48 39 cb             	cmp    %rcx,%rbx
     dbc:	0f 86 5f 01 00 00    	jbe    f21 <MedianPartition+0x2a1>
     dc2:	41 8d 47 01          	lea    0x1(%r15),%eax
     dc6:	49 01 cd             	add    %rcx,%r13
     dc9:	48 01 4c 24 18       	add    %rcx,0x18(%rsp)
     dce:	48 01 4c 24 20       	add    %rcx,0x20(%rsp)
     dd3:	48 29 cb             	sub    %rcx,%rbx
     dd6:	48 29 4c 24 28       	sub    %rcx,0x28(%rsp)
     ddb:	88 44 24 17          	mov    %al,0x17(%rsp)
     ddf:	44 38 64 24 17       	cmp    %r12b,0x17(%rsp)
     de4:	0f 86 41 ff ff ff    	jbe    d2b <MedianPartition+0xab>
     dea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
     df0:	41 0f b6 5d 00       	movzbl 0x0(%r13),%ebx
     df5:	0f 31                	rdtsc  
     df7:	49 89 d4             	mov    %rdx,%r12
     dfa:	4c 8b 44 24 40       	mov    0x40(%rsp),%r8
     dff:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
     e04:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
     e09:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
     e0e:	49 c1 e4 20          	shl    $0x20,%r12
     e12:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
     e17:	89 c0                	mov    %eax,%eax
     e19:	48 89 2c 24          	mov    %rbp,(%rsp)
     e1d:	44 0f b6 cb          	movzbl %bl,%r9d
     e21:	49 09 c4             	or     %rax,%r12
     e24:	e8 d7 f1 ff ff       	callq  0 <Partition>
     e29:	49 89 c5             	mov    %rax,%r13
     e2c:	0f 31                	rdtsc  
     e2e:	48 c1 e2 20          	shl    $0x20,%rdx
     e32:	89 c0                	mov    %eax,%eax
     e34:	48 09 c2             	or     %rax,%rdx
     e37:	4c 39 6c 24 38       	cmp    %r13,0x38(%rsp)
     e3c:	0f 83 8e 01 00 00    	jae    fd0 <MedianPartition+0x350>
     e42:	4c 29 e2             	sub    %r12,%rdx
     e45:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
     e4a:	48 01 55 10          	add    %rdx,0x10(%rbp)
     e4e:	48 01 45 18          	add    %rax,0x18(%rbp)
     e52:	84 db                	test   %bl,%bl
     e54:	75 36                	jne    e8c <MedianPartition+0x20c>
     e56:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
     e5b:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
     e60:	38 1c 38             	cmp    %bl,(%rax,%rdi,1)
     e63:	0f 85 03 01 00 00    	jne    f6c <MedianPartition+0x2ec>
     e69:	48 83 c4 68          	add    $0x68,%rsp
     e6d:	48 89 f8             	mov    %rdi,%rax
     e70:	5b                   	pop    %rbx
     e71:	5d                   	pop    %rbp
     e72:	41 5c                	pop    %r12
     e74:	41 5d                	pop    %r13
     e76:	41 5e                	pop    %r14
     e78:	41 5f                	pop    %r15
     e7a:	c3                   	retq   
     e7b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
     e80:	45 8d 67 ff          	lea    -0x1(%r15),%r12d
     e84:	48 89 cb             	mov    %rcx,%rbx
     e87:	e9 94 fe ff ff       	jmpq   d20 <MedianPartition+0xa0>
     e8c:	0f 31                	rdtsc  
     e8e:	49 89 d4             	mov    %rdx,%r12
     e91:	44 8d 4b ff          	lea    -0x1(%rbx),%r9d
     e95:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
     e9a:	48 8b 54 24 48       	mov    0x48(%rsp),%rdx
     e9f:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
     ea4:	49 c1 e4 20          	shl    $0x20,%r12
     ea8:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
     ead:	89 c0                	mov    %eax,%eax
     eaf:	48 89 2c 24          	mov    %rbp,(%rsp)
     eb3:	45 0f b6 c9          	movzbl %r9b,%r9d
     eb7:	4d 89 e8             	mov    %r13,%r8
     eba:	49 09 c4             	or     %rax,%r12
     ebd:	e8 3e f1 ff ff       	callq  0 <Partition>
     ec2:	48 39 44 24 38       	cmp    %rax,0x38(%rsp)
     ec7:	0f 82 b8 00 00 00    	jb     f85 <MedianPartition+0x305>
     ecd:	0f 31                	rdtsc  
     ecf:	48 8b 4d 10          	mov    0x10(%rbp),%rcx
     ed3:	48 c1 e2 20          	shl    $0x20,%rdx
     ed7:	89 c0                	mov    %eax,%eax
     ed9:	48 09 c2             	or     %rax,%rdx
     edc:	4c 01 6d 18          	add    %r13,0x18(%rbp)
     ee0:	4c 29 e1             	sub    %r12,%rcx
     ee3:	48 01 ca             	add    %rcx,%rdx
     ee6:	48 89 55 10          	mov    %rdx,0x10(%rbp)
     eea:	e9 67 ff ff ff       	jmpq   e56 <MedianPartition+0x1d6>
     eef:	b9 00 00 00 00       	mov    $0x0,%ecx
     ef4:	ba eb 02 00 00       	mov    $0x2eb,%edx
     ef9:	be 00 00 00 00       	mov    $0x0,%esi
     efe:	bf 00 00 00 00       	mov    $0x0,%edi
     f03:	e8 00 00 00 00       	callq  f08 <MedianPartition+0x288>
     f08:	b9 00 00 00 00       	mov    $0x0,%ecx
     f0d:	ba dc 02 00 00       	mov    $0x2dc,%edx
     f12:	be 00 00 00 00       	mov    $0x0,%esi
     f17:	bf 00 00 00 00       	mov    $0x0,%edi
     f1c:	e8 00 00 00 00       	callq  f21 <MedianPartition+0x2a1>
     f21:	b9 00 00 00 00       	mov    $0x0,%ecx
     f26:	ba f4 02 00 00       	mov    $0x2f4,%edx
     f2b:	be 00 00 00 00       	mov    $0x0,%esi
     f30:	bf 00 00 00 00       	mov    $0x0,%edi
     f35:	e8 00 00 00 00       	callq  f3a <MedianPartition+0x2ba>
     f3a:	b9 00 00 00 00       	mov    $0x0,%ecx
     f3f:	ba cd 02 00 00       	mov    $0x2cd,%edx
     f44:	be 00 00 00 00       	mov    $0x0,%esi
     f49:	bf 00 00 00 00       	mov    $0x0,%edi
     f4e:	e8 00 00 00 00       	callq  f53 <MedianPartition+0x2d3>
     f53:	b9 00 00 00 00       	mov    $0x0,%ecx
     f58:	ba cc 02 00 00       	mov    $0x2cc,%edx
     f5d:	be 00 00 00 00       	mov    $0x0,%esi
     f62:	bf 00 00 00 00       	mov    $0x0,%edi
     f67:	e8 00 00 00 00       	callq  f6c <MedianPartition+0x2ec>
     f6c:	b9 00 00 00 00       	mov    $0x0,%ecx
     f71:	ba 37 03 00 00       	mov    $0x337,%edx
     f76:	be 00 00 00 00       	mov    $0x0,%esi
     f7b:	bf 00 00 00 00       	mov    $0x0,%edi
     f80:	e8 00 00 00 00       	callq  f85 <MedianPartition+0x305>
     f85:	b9 00 00 00 00       	mov    $0x0,%ecx
     f8a:	ba 30 03 00 00       	mov    $0x330,%edx
     f8f:	be 00 00 00 00       	mov    $0x0,%esi
     f94:	bf 00 00 00 00       	mov    $0x0,%edi
     f99:	e8 00 00 00 00       	callq  f9e <MedianPartition+0x31e>
     f9e:	b9 00 00 00 00       	mov    $0x0,%ecx
     fa3:	ba cb 02 00 00       	mov    $0x2cb,%edx
     fa8:	be 00 00 00 00       	mov    $0x0,%esi
     fad:	bf 00 00 00 00       	mov    $0x0,%edi
     fb2:	e8 00 00 00 00       	callq  fb7 <MedianPartition+0x337>
     fb7:	b9 00 00 00 00       	mov    $0x0,%ecx
     fbc:	ba ca 02 00 00       	mov    $0x2ca,%edx
     fc1:	be 00 00 00 00       	mov    $0x0,%esi
     fc6:	bf 00 00 00 00       	mov    $0x0,%edi
     fcb:	e8 00 00 00 00       	callq  fd0 <MedianPartition+0x350>
     fd0:	b9 00 00 00 00       	mov    $0x0,%ecx
     fd5:	ba 1c 03 00 00       	mov    $0x31c,%edx
     fda:	be 00 00 00 00       	mov    $0x0,%esi
     fdf:	bf 00 00 00 00       	mov    $0x0,%edi
     fe4:	e8 00 00 00 00       	callq  fe9 <MedianPartition+0x369>
     fe9:	b9 00 00 00 00       	mov    $0x0,%ecx
     fee:	ba ce 02 00 00       	mov    $0x2ce,%edx
     ff3:	be 00 00 00 00       	mov    $0x0,%esi
     ff8:	bf 00 00 00 00       	mov    $0x0,%edi
     ffd:	e8 00 00 00 00       	callq  1002 <MedianPartition+0x382>
    1002:	0f 1f 40 00          	nopl   0x0(%rax)
    1006:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    100d:	00 00 00 

0000000000001010 <MPWorkspaceInit>:
    1010:	48 85 ff             	test   %rdi,%rdi
    1013:	53                   	push   %rbx
    1014:	48 89 fb             	mov    %rdi,%rbx
    1017:	74 1b                	je     1034 <MPWorkspaceInit+0x24>
    1019:	48 85 f6             	test   %rsi,%rsi
    101c:	74 2f                	je     104d <MPWorkspaceInit+0x3d>
    101e:	48 c1 ee 05          	shr    $0x5,%rsi
    1022:	48 8d 3c f5 00 00 00 	lea    0x0(,%rsi,8),%rdi
    1029:	00 
    102a:	e8 00 00 00 00       	callq  102f <MPWorkspaceInit+0x1f>
    102f:	48 89 03             	mov    %rax,(%rbx)
    1032:	5b                   	pop    %rbx
    1033:	c3                   	retq   
    1034:	b9 00 00 00 00       	mov    $0x0,%ecx
    1039:	ba 41 03 00 00       	mov    $0x341,%edx
    103e:	be 00 00 00 00       	mov    $0x0,%esi
    1043:	bf 00 00 00 00       	mov    $0x0,%edi
    1048:	e8 00 00 00 00       	callq  104d <MPWorkspaceInit+0x3d>
    104d:	b9 00 00 00 00       	mov    $0x0,%ecx
    1052:	ba 42 03 00 00       	mov    $0x342,%edx
    1057:	be 00 00 00 00       	mov    $0x0,%esi
    105c:	bf 00 00 00 00       	mov    $0x0,%edi
    1061:	e8 00 00 00 00       	callq  1066 <MPWorkspaceInit+0x56>
    1066:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    106d:	00 00 00 

0000000000001070 <MPWorkspaceDestroy>:
    1070:	48 85 ff             	test   %rdi,%rdi
    1073:	74 08                	je     107d <MPWorkspaceDestroy+0xd>
    1075:	48 8b 3f             	mov    (%rdi),%rdi
    1078:	e9 00 00 00 00       	jmpq   107d <MPWorkspaceDestroy+0xd>
    107d:	50                   	push   %rax
    107e:	b9 00 00 00 00       	mov    $0x0,%ecx
    1083:	ba 4b 03 00 00       	mov    $0x34b,%edx
    1088:	be 00 00 00 00       	mov    $0x0,%esi
    108d:	bf 00 00 00 00       	mov    $0x0,%edi
    1092:	e8 00 00 00 00       	callq  1097 <MPWorkspaceDestroy+0x27>
