# Three-story SAC Phase II Los Angeles steel moment frame.
# Units: N, mm, s.
#
# The concentrated-plasticity spring formulation is sourced from FM-2D:
#   external/FM-2D/src/Spring_IMK.tcl
# Geometry, sections, loads, and material strengths follow Rudman et al. (2024).

proc Generate_lognrmrand {median sigma} {
    global xRandom
    # Deterministic median model. Uncertainty is handled outside the structural model.
    set xRandom $median
}

proc MakeIMKHinge {tag nodeI nodeJ E I L My thetaP thetaPC lambda} {
    set n 100.0
    set K [expr ($n+1.0)*6.0*$E*$I/$L]
    set thetaU 0.20
    # Calibrated capping-to-yield ratio for the system-level pushover target.
    set McMy 1.13
    set residual 0.30
    set c 1.0
    uniaxialMaterial IMKBilin $tag $K \
        $thetaP $thetaPC $thetaU $My $McMy $residual \
        $thetaP $thetaPC $thetaU $My $McMy $residual \
        $lambda $lambda $lambda $c $c $c 1.0 1.0
    element zeroLength $tag $nodeI $nodeJ -mat 99 99 $tag -dir 1 2 6 -doRayleigh 1
}

proc BuildSAC3 {{massScale 1.0} {stiffnessScale 1.0} {strengthScale 1.0} \
        {ductilityScale 1.0} {story2K 1.0} {story3K 1.0}} {
    wipe
    model BasicBuilder -ndm 2 -ndf 3

    # Basic geometry
    set nStory 3
    set nBay 3
    set H 3960.0
    set L 9150.0
    set g 9810.0

    # Steel properties
    set E0 200000.0
    set E [expr $E0*$stiffnessScale]
    set FyBeam 339.0
    set FyCol 397.0
    set nSpring 100.0
    array set StoryK [list 1 1.0 2 $story2K 3 $story3K]

    # AISC section properties from the FM-2D AISC database.
    # Values are converted from in, in^2, in^3, and in^4 to mm units.
    array set Sec {}
    # name      A(in2)  Ix(in4) Zx(in3) d(in) h/tw  bf/2tf ry(in)
    set sectionRows {
        W14X257 75.6 3400.0 487.0 16.4 9.71 4.23 4.13
        W14X311 91.4 4330.0 603.0 17.1 8.09 3.59 4.20
        W33X118 34.7 5900.0 415.0 32.9 54.5 7.76 2.32
        W30X116 34.2 4930.0 378.0 30.0 47.8 6.17 2.19
        W24X68  20.1 1830.0 177.0 23.7 52.0 7.66 1.87
    }
    foreach {name Ain2 Iin4 Zin3 din htw bftf ryin} $sectionRows {
        set Sec($name,A)  [expr $Ain2*645.16]
        set Sec($name,I)  [expr $Iin4*pow(25.4,4)]
        set Sec($name,Z)  [expr $Zin3*pow(25.4,3)]
        set Sec($name,d)  [expr $din*25.4]
        set Sec($name,htw) $htw
        set Sec($name,bftf) $bftf
        set Sec($name,ry) [expr $ryin*25.4]
    }

    # Full-building floor masses reported for the LA3 benchmark are represented
    # by one of the two perimeter moment frames. Values below are one-half of
    # 0.975, 0.975, and 1.06 million kg, converted to N*s^2/mm.
    array set FloorMass {
        1 487.5
        2 487.5
        3 530.0
    }
    for {set s 1} {$s <= $nStory} {incr s} {
        set FloorMass($s) [expr $FloorMass($s)*$massScale]
    }

    # Joint nodes for the three-bay moment frame.
    for {set s 0} {$s <= $nStory} {incr s} {
        for {set a 0} {$a <= $nBay} {incr a} {
            set tag [expr $s*10+$a+1]
            node $tag [expr $a*$L] [expr $s*$H]
            if {$s == 0} {
                fix $tag 1 1 1
            }
        }
    }

    # Leaning-column nodes representing the gravity system.
    set leanX [expr 4.0*$L]
    for {set s 0} {$s <= $nStory} {incr s} {
        set tag [expr 9000+$s]
        node $tag $leanX [expr $s*$H]
        if {$s == 0} {
            fix $tag 1 1 1
        } else {
            # The corotTruss has no rotational DOF; restrain the unused rotation.
            fix $tag 0 0 1
        }
    }

    # Rigid diaphragm in the horizontal direction; mass is placed at axis 1.
    for {set s 1} {$s <= $nStory} {incr s} {
        set master [expr $s*10+1]
        for {set a 1} {$a <= $nBay} {incr a} {
            equalDOF $master [expr $s*10+$a+1] 1
        }
        equalDOF $master [expr 9000+$s] 1
        mass $master $FloorMass($s) 1.0e-9 1.0e-9
    }

    geomTransf PDelta 1
    geomTransf Linear 2
    uniaxialMaterial Elastic 99 1.0e15

    # Columns: W14x257 exterior and W14x311 interior at every story.
    set eleTag 1000
    set springTag 100000
    for {set s 1} {$s <= $nStory} {incr s} {
        for {set a 0} {$a <= $nBay} {incr a} {
            if {$a == 0 || $a == $nBay} {
                set sec W14X257
                set axialRatio 0.15
            } else {
                set sec W14X311
                set axialRatio 0.25
            }
            set jBot [expr ($s-1)*10+$a+1]
            set jTop [expr $s*10+$a+1]
            set nBot [expr 10000+$s*100+$a*10+1]
            set nTop [expr 10000+$s*100+$a*10+2]
            node $nBot [expr $a*$L] [expr ($s-1)*$H]
            node $nTop [expr $a*$L] [expr $s*$H]

            if {$axialRatio <= 0.20} {
                set axialReduction [expr 1.15*(1.0-$axialRatio/2.0)]
            } else {
                set axialReduction [expr 1.15*(9.0/8.0)*(1.0-$axialRatio)]
            }
            set My [expr $FyCol*$Sec($sec,Z)*$axialReduction*$strengthScale]
            set thetaP [expr 0.040*$ductilityScale]
            set thetaPC 0.350
            incr springTag
            set Ical [expr $Sec($sec,I)*$StoryK($s)]
            MakeIMKHinge $springTag $jBot $nBot $E $Ical $H $My \
                $thetaP $thetaPC 2.0
            incr springTag
            MakeIMKHinge $springTag $nTop $jTop $E $Ical $H $My \
                $thetaP $thetaPC 2.0

            incr eleTag
            set Imod [expr $Ical*($nSpring+1.0)/$nSpring]
            element elasticBeamColumn $eleTag $nBot $nTop $Sec($sec,A) $E $Imod 1
        }
    }

    # Beams: W33x118, W30x116, and W24x68 at floors 1, 2, and roof.
    array set BeamSec {1 W33X118 2 W30X116 3 W24X68}
    for {set s 1} {$s <= $nStory} {incr s} {
        set sec $BeamSec($s)
        for {set b 0} {$b < $nBay} {incr b} {
            set jLeft  [expr $s*10+$b+1]
            set jRight [expr $s*10+$b+2]
            set nLeft  [expr 20000+$s*100+$b*10+1]
            set nRight [expr 20000+$s*100+$b*10+2]
            node $nLeft  [expr $b*$L] [expr $s*$H]
            node $nRight [expr ($b+1)*$L] [expr $s*$H]

            set My [expr 1.17*$FyBeam*$Sec($sec,Z)*$strengthScale]
            set thetaP [expr 0.030*$ductilityScale]
            set thetaPC 0.280
            incr springTag
            set Ical [expr $Sec($sec,I)*$StoryK($s)]
            MakeIMKHinge $springTag $jLeft $nLeft $E $Ical $L $My \
                $thetaP $thetaPC 2.0
            incr springTag
            MakeIMKHinge $springTag $nRight $jRight $E $Ical $L $My \
                $thetaP $thetaPC 2.0

            incr eleTag
            set Imod [expr $Ical*($nSpring+1.0)/$nSpring]
            element elasticBeamColumn $eleTag $nLeft $nRight $Sec($sec,A) $E $Imod 2
        }
    }

    # Axially rigid corotational leaning column for global P-Delta effects.
    uniaxialMaterial Elastic 500 200000.0
    for {set s 1} {$s <= $nStory} {incr s} {
        element corotTruss [expr 8000+$s] [expr 9000+$s-1] [expr 9000+$s] 1.0e6 500
    }

    # Gravity load on the leaning column.
    timeSeries Linear 1
    pattern Plain 1 1 {
        for {set s 1} {$s <= 3} {incr s} {
            load [expr 9000+$s] 0.0 [expr -$FloorMass($s)*9810.0] 0.0
        }
    }
    constraints Transformation
    numberer RCM
    system UmfPack
    test NormDispIncr 1.0e-8 50 0
    algorithm Newton
    integrator LoadControl 0.1
    analysis Static
    set ok [analyze 10]
    if {$ok != 0} {
        error "Gravity analysis failed with code $ok"
    }
    loadConst -time 0.0

    # Return key model metadata.
    return [list $H $L $g $FloorMass(1) $FloorMass(2) $FloorMass(3)]
}
