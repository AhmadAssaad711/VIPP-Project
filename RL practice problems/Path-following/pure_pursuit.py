import pygame
import math
import sys
import numpy as np

# ---------- Config ----------
WIDTH, HEIGHT = 1000, 600
FPS = 60

SCALE = 20
DT = 0.05
SPEED = 0.5
WHEEL_BASE = 2.65
LOOKAHEAD = 5.0

# ---------- Helpers ----------
def world_to_screen(x, y):
    return int(x * SCALE), HEIGHT - int(y * SCALE)

def global_to_vehicle(px, py, car):
    dx = px - car.x
    dy = py - car.y
    c = math.cos(-car.yaw)
    s = math.sin(-car.yaw)
    return c*dx - s*dy, s*dx + c*dy

def circle_line_segment_intersection(radius, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dr2 = dx*dx + dy*dy
    D = x1*y2 - x2*y1
    disc = radius*radius*dr2 - D*D

    if disc < 0:
        return []

    sqrt_disc = math.sqrt(disc)
    sign_dy = 1 if dy >= 0 else -1
    pts = []

    for s in [1, -1]:
        px = (D*dy + s*sign_dy*dx*sqrt_disc) / dr2
        py = (-D*dx + s*abs(dy)*sqrt_disc) / dr2

        # compute t
        if abs(dx) > abs(dy):
            t = (px - x1) / dx
        else:
            t = (py - y1) / dy

        if 0 <= t <= 1:
            pts.append((px, py))

    return pts

def get_target_point(local_path):
    for i in range(len(local_path) - 1):
        hits = circle_line_segment_intersection(
            LOOKAHEAD, local_path[i], local_path[i+1]
        )
        hits = [p for p in hits if p[0] > 0]  # forward only
        if hits:
            return hits[0]
    return None

# ---------- Vehicle ----------
class Vehicle:
    def __init__(self, x=2.0, y=2.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

    def step(self, delta):
        self.x += SPEED * math.cos(self.yaw) * DT
        self.y += SPEED * math.sin(self.yaw) * DT
        self.yaw += (SPEED / WHEEL_BASE) * math.tan(delta) * DT

# ---------- Main ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Step 5: Pure Pursuit Steering")
    clock = pygame.time.Clock()

    car = Vehicle()

    path = [(i, 5 + 2 * math.sin(0.3 * i)) for i in np.linspace(0, 40, 300)]

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # --- Transform path ---
        local_path = [global_to_vehicle(px, py, car) for px, py in path]

        # --- Pure Pursuit ---
        delta = 0.0
        target = get_target_point(local_path)
        if target:
            alpha = math.atan2(target[1], target[0])
            delta = math.atan2(
                2 * WHEEL_BASE * math.sin(alpha),
                LOOKAHEAD
            )

        # --- Physics ---
        car.step(delta)

        # --- Draw ---
        screen.fill((30, 30, 30))

        # Path
        for i in range(len(path) - 1):
            pygame.draw.line(
                screen, (180, 180, 180),
                world_to_screen(*path[i]),
                world_to_screen(*path[i+1]), 2
            )

        # Vehicle
        cx, cy = world_to_screen(car.x, car.y)
        pygame.draw.circle(screen, (0, 255, 0), (cx, cy), 6)

        hx = cx + int(20 * math.cos(car.yaw))
        hy = cy - int(20 * math.sin(car.yaw))
        pygame.draw.line(screen, (0, 255, 0), (cx, cy), (hx, hy), 2)

        # Lookahead circle
        pygame.draw.circle(
            screen, (100, 100, 255),
            (cx, cy), int(LOOKAHEAD * SCALE), 1
        )

        # Target
        if target:
            tx = car.x + math.cos(car.yaw)*target[0] - math.sin(car.yaw)*target[1]
            ty = car.y + math.sin(car.yaw)*target[0] + math.cos(car.yaw)*target[1]
            pygame.draw.circle(screen, (255, 0, 0), world_to_screen(tx, ty), 6)

        pygame.display.flip()

if __name__ == "__main__":
    main()
