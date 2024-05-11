import openmc
import openmc.deplete
import numpy as np
from math import sqrt
import pandas as pd
from mpi4py import MPI

FuelSalt =openmc.Material(name='FuelSalt')
FuelSalt.add_nuclide('U234', 0.000528988, 'wo')
FuelSalt.add_nuclide('U235', 0.05072890, 'wo')
FuelSalt.add_nuclide('U236', 0.000242995, 'wo')
FuelSalt.add_nuclide('U238', 0.205354, 'wo')
FuelSalt.add_nuclide('Li6', 0.0000089998, 'wo')
FuelSalt.add_nuclide('Li7', 0.101955, 'wo')
FuelSalt.add_nuclide('F19',0.586948, 'wo')
FuelSalt.add_nuclide('Be9', 0.0542328, 'wo')
FuelSalt.set_density('g/cc', 2.645)
FuelSalt.volume = 391291
FuelSalt.temperature = 873

graphite = openmc.Material(name='graphite')
graphite.add_element('B', 0.000001, 'wo')
graphite.add_nuclide('C0', 0.999999, 'wo')
graphite.set_density('g/cc', 1.8)
graphite.add_s_alpha_beta('c_Graphite')
graphite.temperature = 873

air=openmc.Material(name='air')
air.add_nuclide('C0', 0.000124, 'wo')
air.add_nuclide('N14' ,0.755268, 'wo') 
air.add_nuclide('O16', 0.231781, 'wo')
air.add_element('Ar', 0.012827, 'wo')
air.set_density('g/cc', 1.29E-3)
air.temperature = 300

m1conc=openmc.Material(name='m1')
m1conc.add_nuclide('H1', 0.008, 'wo')
m1conc.add_element('B', 0.009, 'wo')
m1conc.add_nuclide('O16', 0.107, 'wo')
m1conc.add_element('Mg', 0.043, 'wo')
m1conc.add_element('Cl', 0.021, 'wo')
m1conc.add_nuclide('Mn55', 0.03, 'wo')
m1conc.add_element('Ca', 0.011, 'wo')
m1conc.add_element('Fe', 0.798, 'wo')
m1conc.set_density('g/cc', 4.5)
m1conc.temperature = 300

poly=openmc.Material(name='poly')
poly.add_nuclide('H1', 0.143716, 'wo')
poly.add_nuclide('C0', 0.856284, 'wo')
poly.set_density('g/cc', 0.94)
poly.temperature = 300

b4c=openmc.Material(name='b4c')
b4c.add_element('B', 0.782610, 'wo')
b4c.add_nuclide('C0', 0.217390, 'wo')
b4c.set_density('g/cc', 1.89)
b4c.temperature = 300

OakRidgeConc=openmc.Material(name='OakRidgeConc')
OakRidgeConc.add_nuclide('H1', 0.006187, 'wo')
OakRidgeConc.add_nuclide('C0', 0.175193, 'wo')
OakRidgeConc.add_nuclide('O16', 0.410184, 'wo')
OakRidgeConc.add_nuclide('Na23', 0.00271, 'wo')
OakRidgeConc.add_element('Mg', 0.032649, 'wo')
OakRidgeConc.add_nuclide('Al27', 0.0183, 'wo')
OakRidgeConc.add_element('Si', 0.034479, 'wo')
OakRidgeConc.add_element('K', 0.001138, 'wo')
OakRidgeConc.add_element('Ca', 0.321287, 'wo')
OakRidgeConc.add_element('Fe', 0.7784, 'wo')
OakRidgeConc.set_density('g/cc', 2.3)
OakRidgeConc.temperature = 300

SS316H=openmc.Material(name='SS316H')
SS316H.add_nuclide('C0',0.001, 'wo')
SS316H.add_element('Si', 0.00507, 'wo')
SS316H.add_nuclide('P31', 0.0023, 'wo')
SS316H.add_element('S', 0.00015, 'wo')
SS316H.add_element('Cr', 0.17, 'wo')
SS316H.add_nuclide('Mn55', 0.01014, 'wo')
SS316H.add_element('Fe', 0.66841, 'wo')
SS316H.add_element('Ni', 0.12, 'wo')
SS316H.add_element('Mo', 0.025, 'wo')
SS316H.set_density('g/cc', 8)
SS316H.temperature = 600

#was gold schottky, changed to Nickel
Au = openmc.Material(name='Ni')
Au.add_element('Ni', 1.0)
Au.set_density('g/cc', 8.9)
Au.temperature = 300

SiC = openmc.Material(name='SiC')
SiC.add_element('Si', 0.5)
SiC.add_nuclide('C0', 0.5)
SiC.set_density('g/cc', 3.22)
SiC.temperature = 300

Li = openmc.Material(name='Li')
Li.add_nuclide('Li6', .95, 'wo')
Li.add_nuclide('Li7', .05, 'wo')
Li.set_density('g/cc', 2.64)
Li.temperature = 300

F = openmc.Material(name='F')
F.add_element('F', 1)
F.set_density('g/cc', 2.64)
F.temperature = 300

LiF=openmc.Material.mix_materials([Li, F], [.5, .5], 'ao')

# Instantiate a Materials collection and export to xml
materials_file = openmc.Materials([FuelSalt, graphite, air, m1conc, poly, b4c, OakRidgeConc, SS316H, Au, SiC, LiF])
materials_file.export_to_xml()

# Detector = openmc.XCylinder(y0=0.0, z0=0.0, r=0.124122)

#define regions
fuel_channel=openmc.ZCylinder(r=1.508)
fuel_cell = openmc.Cell(fill=FuelSalt, region=-fuel_channel)
graphite_cell = openmc.Cell(fill=graphite, region=+fuel_channel)
fuel_universe = openmc.Universe(cells=(fuel_cell, graphite_cell))

graphite_cell2=openmc.Cell(fill=graphite)
outer_universe = openmc.Universe(cells=(graphite_cell2,))

lattice = openmc.HexLattice()

lattice.center = (0,0)
lattice.pitch = (10.16,)
lattice.outer = outer_universe

ring_6 = [fuel_universe]*5+[outer_universe]+[fuel_universe]*4+[outer_universe]+[fuel_universe]*9+[outer_universe]+[fuel_universe]*4+[outer_universe]+[fuel_universe]*4
ring_5 = [fuel_universe]*24
ring_4 = [fuel_universe]*18
ring_3 = [fuel_universe]*12
ring_2 = [fuel_universe]*6
ring_1 = [fuel_universe]*1

core_top = openmc.ZPlane(z0=+151.68)
core_bot = openmc.ZPlane(z0=0)

lattice.universes = [ring_6, ring_5, ring_4, ring_3, ring_2, ring_1]
lattice_boundary = openmc.ZCylinder(r=64)
reactor_cell = openmc.Cell(fill=lattice, region=-lattice_boundary & -core_top & +core_bot)

annulus_boundary = openmc.ZCylinder(r=65)

annulus_cell = openmc.Cell(fill=FuelSalt, region=+lattice_boundary & -annulus_boundary & -core_top & +core_bot)

vessel_boundary = openmc.ZCylinder(r=67)
vessel_cell = openmc.Cell(fill=SS316H, region=+annulus_boundary & -vessel_boundary & -core_top & +core_bot)

top_fuel_hemisphere = openmc.Sphere(z0=38.475, r=130.54)
top_vessel_hemisphere = openmc.Sphere(z0=38.475, r=131.5425)
top_fuel_hemisphere_cell = openmc.Cell(fill=FuelSalt, region=-top_fuel_hemisphere & +core_top)
top_vessel_hemisphere_cell = openmc.Cell(fill=SS316H, region=-top_vessel_hemisphere &+top_fuel_hemisphere & +core_top)

bot_fuel_hemisphere = openmc.Sphere(z0=113.205, r=130.54)
bot_vessel_hemisphere = openmc.Sphere(z0=113.205, r=131.5425)
bot_fuel_hemisphere_cell = openmc.Cell(fill=FuelSalt, region=-bot_fuel_hemisphere & -core_bot)
bot_vessel_hemisphere_cell = openmc.Cell(fill=SS316H, region=-bot_vessel_hemisphere &+bot_fuel_hemisphere & -core_bot)

air_top = openmc.ZPlane(z0=+200)
air_bot = openmc.ZPlane(z0=-150)

detectorx0=69
detectorx1=detectorx0+.1
detectorx2=detectorx1+0.0008
detectorx3=detectorx2+.015
detectorx4=detectorx3+.035
detectorx5=detectorx4+.0008

Lix0 = openmc.XPlane(x0=detectorx0)
Goldx1 = openmc.XPlane(x0=detectorx1)
Goldx2 = openmc.XPlane(x0=detectorx2)
SiCx3 = openmc.XPlane(x0=detectorx3)
SiCx4 = openmc.XPlane(x0=detectorx4)
Goldx5 = openmc.XPlane(x0=detectorx5)
Detector = openmc.XCylinder(y0=0.0, z0=75.0, r=.2481)

Lithium_cell = openmc.Cell(name='Lithium')
Lithium_cell.fill = LiF
Lithium_cell.region = -Detector & -Goldx1 & +Lix0 


Gold_cell = openmc.Cell(name='Gold')
Gold_cell.fill = Au
Gold_cell.region = -Detector & -Goldx2 & +Goldx1 

SiC_cell = openmc.Cell(name='SiC')
SiC_cell.fill = SiC
SiC_cell.region = +Goldx2 & -SiCx3 & -Detector

Inactive_SiC_cell = openmc.Cell(name='SiC2')
Inactive_SiC_cell.fill = SiC
Inactive_SiC_cell.region = +SiCx3 & -SiCx4 & -Detector

Gold_cell2 = openmc.Cell(name='Gold')
Gold_cell2.fill = Au
Gold_cell2.region = -Detector & -Goldx5 & +SiCx4 

internal_shield_inner = openmc.ZCylinder(r=122)
air1_cell = openmc.Cell(fill=air, region=+vessel_boundary & -internal_shield_inner & +air_bot & -air_top &+Detector)

detectorair1_cell = openmc.Cell(fill=air, region=+vessel_boundary & -internal_shield_inner & +air_bot & -air_top &-Detector &-Lix0)
detectorair2_cell = openmc.Cell(fill=air, region=+vessel_boundary & -internal_shield_inner & +air_bot & -air_top &-Detector &+Goldx5)

air4_cell = openmc.Cell(fill=air, region=-vessel_boundary & +top_vessel_hemisphere & +core_top & -air_top)

air5_cell = openmc.Cell(fill=air, region=-vessel_boundary & +bot_vessel_hemisphere & +air_bot & -core_bot)

poly_top = openmc.ZPlane(z0=+212.7)
poly_bot = openmc.ZPlane(z0=-162.7)

internal_shield_mid = openmc.ZCylinder(r=134.7)
poly_cell = openmc.Cell(fill=poly, region=+internal_shield_inner & -internal_shield_mid &+poly_bot & -poly_top)

BC_top = openmc.ZPlane(z0=+213.97)
BC_bot = openmc.ZPlane(z0=-163.97)

poly_cell2 = openmc.Cell(fill=poly, region=-internal_shield_mid &+air_top & -poly_top)
poly_cell3 = openmc.Cell(fill=poly, region=-internal_shield_mid &-air_bot & +poly_bot)

internal_shield_outer = openmc.ZCylinder(r=135.97)
BC_cell = openmc.Cell(fill=b4c, region=+internal_shield_mid & -internal_shield_outer &+BC_bot & -BC_top)

air2_top = openmc.ZPlane(z0=+300)
air2_bot = openmc.ZPlane(z0=-180.54)

BC_cell2 = openmc.Cell(fill=b4c, region=-internal_shield_outer &+poly_top & -BC_top)
BC_cell3 = openmc.Cell(fill=b4c, region=-internal_shield_outer &-poly_bot & +BC_bot)

reactor_enclosure_inner=openmc.ZCylinder(r=152.54)
air2_cell = openmc.Cell(fill=air, region=+internal_shield_outer & -reactor_enclosure_inner &+air2_bot & -air2_top)

enclosure_top = openmc.ZPlane(z0=+302.54)
enclosure_bot = openmc.ZPlane(z0=-183.08)

air2_cell2 = openmc.Cell(fill=air, region=-reactor_enclosure_inner &+BC_top & -air2_top)
air2_cell3 = openmc.Cell(fill=air, region=-reactor_enclosure_inner &-BC_bot & +air2_bot)

reactor_enclosure_outer=openmc.ZCylinder(r=155.08)
reactor_enclosure_cell = openmc.Cell(fill=SS316H, region=+reactor_enclosure_inner & -reactor_enclosure_outer &+enclosure_bot & -enclosure_top)

air3_top = openmc.ZPlane(z0=+362.54)
air3_bot = openmc.ZPlane(z0=-213.08)

reactor_enclosure_cell2 = openmc.Cell(fill=SS316H, region=-reactor_enclosure_outer &+air2_top & -enclosure_top)
reactor_enclosure_cell3 = openmc.Cell(fill=SS316H, region=-reactor_enclosure_outer &-air2_bot & +enclosure_bot)

borated_concrete_inner = openmc.ZCylinder(r=167.64)
air3_cell = openmc.Cell(fill=air, region=+reactor_enclosure_outer & -borated_concrete_inner &+air3_bot & -air3_top)

concrete_shield_top = openmc.ZPlane(z0=+423.54)
concrete_shield_bot = openmc.ZPlane(z0=-274.08)

air3_cell2 = openmc.Cell(fill=air, region=-borated_concrete_inner &+enclosure_top & -air3_top)
air3_cell3 = openmc.Cell(fill=air, region=-borated_concrete_inner &-enclosure_bot & +air3_bot)

borated_concrete_outer = openmc.ZCylinder(r=228.6)
borated_concrete_cell = openmc.Cell(fill=m1conc, region=+borated_concrete_inner & -borated_concrete_outer &+concrete_shield_bot & -concrete_shield_top)

min_z = openmc.ZPlane(z0=-396, boundary_type='vacuum')
max_z = openmc.ZPlane(z0=+515.46, boundary_type='vacuum')

borated_concrete_cell2 = openmc.Cell(fill=m1conc, region=-borated_concrete_outer &+air3_top & -concrete_shield_top)
borated_concrete_cell3 = openmc.Cell(fill=m1conc, region=-borated_concrete_outer &-air3_bot & +concrete_shield_bot)

outer_surface = openmc.ZCylinder(r=350.52, boundary_type='vacuum')
structural_concrete_cell = openmc.Cell(fill=OakRidgeConc, region=+borated_concrete_outer & -outer_surface & +min_z & -max_z)

structural_concrete_cell2 = openmc.Cell(fill=OakRidgeConc, region=-outer_surface & +concrete_shield_top & -max_z)
structural_concrete_cell3 = openmc.Cell(fill=OakRidgeConc, region=-outer_surface & -concrete_shield_bot & +min_z)


detector_universe = openmc.Universe(cells=[Lithium_cell, Gold_cell, SiC_cell, Inactive_SiC_cell, Gold_cell2])

detector_cell = openmc.Cell(fill=detector_universe, region=-Detector & -Goldx5 & +Lix0)


root_universe = openmc.Universe(cells=[reactor_cell, annulus_cell, vessel_cell, top_fuel_hemisphere_cell, top_vessel_hemisphere_cell, bot_fuel_hemisphere_cell, bot_vessel_hemisphere_cell, 
                                       air1_cell, poly_cell, poly_cell2, poly_cell3, BC_cell, BC_cell2, BC_cell3, air2_cell, air2_cell2, air2_cell3,
                                       reactor_enclosure_cell, reactor_enclosure_cell2, reactor_enclosure_cell3, air3_cell, air3_cell2, air3_cell3,
                                       borated_concrete_cell, borated_concrete_cell2, borated_concrete_cell3, structural_concrete_cell, structural_concrete_cell2, structural_concrete_cell3,
                                       air4_cell, air5_cell, detector_cell, detectorair1_cell, detectorair2_cell])



main_cell = openmc.Cell(fill=root_universe, region=-outer_surface & +min_z &-max_z)

geometry = openmc.Geometry([main_cell])
geometry.export_to_xml()

point = openmc.stats.Point((0, 0, 75))
source = openmc.Source(space=point)
settings=openmc.Settings()
settings.temperature = {"method": "interpolation"}
settings.source=source
settings.batches=100
settings.inactive=25
settings.particles=100000
settings.export_to_xml()


#- Coupled Operator and Model

model = openmc.Model(geometry=geometry, settings=settings, materials=materials_file)
operator = openmc.deplete.CoupledOperator(model, "chain_mod_q.xml", fission_yield_mode='average')


d = 24*60*60        # Convert to days to seconds 
power_den = .01 #3.763     # Reactor power density [W/gU]


timesteps = [1, 5, 5, 10, 29, 50, 400, 300, 200, 600, 450]

#- Not cumulative burnup [MWd/kg] or burnup time [s]

integrator = openmc.deplete.CECMIntegrator(operator, timesteps, power_density=power_den,
                                                timestep_units='d')
integrator.integrate()