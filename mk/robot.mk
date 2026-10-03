# Ces cibles nécessitent make setup-udev (une fois) pour créer /dev/lerobot_*

# Les cibles calibrate-follower et calibrate-leader sont à lancer une seule fois pour chaque bras pour lancer la calibration le leader et le follower
calibrate-follower:
	docker compose run --rm lerobot lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=/dev/ttyACM0 \
		--robot.id=follower_arm

calibrate-leader:
	docker compose run --rm lerobot lerobot-calibrate \
		--teleop.type=so101_leader \
		--teleop.port=/dev/ttyACM1 \
		--teleop.id=leader_arm

teleop:
	docker compose run --rm lerobot \
		lerobot-teleoperate \
		--teleop.type=so101_leader \
		--teleop.port=/dev/ttyACM1 \
		--teleop.id=leader_arm \
		--robot.type=so101_follower \
		--robot.port=/dev/ttyACM0 \
		--robot.id=follower_arm

# Vérifie que ttyACM0=follower et ttyACM1=leader dans le container
check-devices:
	docker compose run --rm lerobot python3 scripts/robot/check_devices.py

# Requiert robots branchés, vérifie les ids des servo-moteurs
scan-motors:
	docker compose run --rm lerobot python3 scripts/robot/scan_motors.py

# Vérifie le voltage de chaque servo (un bras à la fois, sans symlink udev)
check-voltage-follower:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower python3 scripts/robot/diag_voltage.py

check-voltage-leader:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM1 lerobot-leader python3 scripts/robot/diag_voltage.py

check-voltage: check-voltage-follower check-voltage-leader

# Lancer un script avec accès direct au follower (sans symlink udev)
# Usage : make script-follower FILE=scripts/robot/move_central_position.py
script-follower:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower python3 $(FILE)

# Lancer un script avec accès direct au leader (sans symlink udev)
# Usage : make script-leader FILE=scripts/robot/move_central_position.py
script-leader:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM1 lerobot-leader python3 $(FILE)
