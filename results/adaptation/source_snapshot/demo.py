"""Local Pygame policy playback. Space: pause; R: reset; arrows: manual moves."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from warehouse.env import WarehouseEnv
from warehouse.agents import QLearning, DQN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/main")
    p.add_argument("--method", choices=["q_learning", "dqn", "random", "manual"], default="q_learning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--headless", action="store_true", help="Save a screenshot of final frame without opening a window")
    p.add_argument("--screenshot", default="results/demo.png")
    args = p.parse_args()
    if args.fps < 1:
        p.error("fps must be positive")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
    out = Path(args.results)
    cfg = json.loads((out / "config.json").read_text())
    mode = "vector" if args.method == "dqn" else "discrete"
    env = WarehouseEnv(cfg["environment"], mode)
    agent = None
    if args.method == "q_learning":
        agent = QLearning(env.observation_space.n, cfg["q_learning"])
        agent.q = np.load(out / args.method / f"seed_{args.seed}" / "q_table.npy")
    elif args.method == "dqn":
        agent = DQN(cfg["dqn"], args.seed)
        agent.load(out / args.method / f"seed_{args.seed}" / "model.pt")
    rng = np.random.default_rng(args.seed)
    pygame.init()
    cell, top = 48, 100
    screen = pygame.display.set_mode((max(env.width*cell, 700), env.height*cell+top+65))
    pygame.display.set_caption("Warehouse Navigation — trained policy playback")
    font = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()
    state, info = env.reset()
    total, paused, running, route = 0.0, False, True, [env.start]
    keys = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state, info = env.reset()
                    total, route = 0.0, [env.start]
                elif args.method == "manual" and event.key in keys:
                    action = keys[event.key]
        if not paused and not env.finished and args.method != "manual":
            action = agent.greedy(state) if agent is not None else int(rng.integers(4))
        if action is not None and not env.finished and not paused:
            state, reward, _, _, info = env.step(action)
            total += reward
            route.append(info["position"])
        screen.fill((248, 249, 251))
        status = "DELIVERED" if info["success"] else "STEP LIMIT" if env.finished else "PAUSED" if paused else "RUNNING"
        texts = [f"{args.method} | seed {args.seed} | {status}",
                 f"Steps {env.steps}/{env.max_steps}   Return {total:.2f}   Collisions {env.collisions}   Carrying {env.carrying}",
                 "Space pause   R reset   Esc quit   Arrows move in manual mode"]
        for i, text in enumerate(texts):
            screen.blit(font.render(text, True, (25, 35, 45)), (12, 12+i*26))
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(x*cell, top+y*cell, cell, cell)
                color = (90, 100, 110) if (x, y) in env.shelves else (235, 239, 243)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (175, 183, 191), rect, 1)
        colors = {env.start: (150, 180, 220), env.pickup: (250, 200, 80), env.delivery: (90, 190, 120)}
        for point, label in ((env.start, "S"), (env.pickup, "P"), (env.delivery, "D")):
            x, y = point
            pygame.draw.rect(screen, colors[point], (x*cell+2, top+y*cell+2, cell-4, cell-4))
            screen.blit(font.render(label, True, (20, 30, 40)), (x*cell+7, top+y*cell+5))
        if len(route) > 1:
            pygame.draw.lines(screen, (100, 150, 210), False, [(x*cell+cell//2, top+y*cell+cell//2) for x, y in route], 3)
        pygame.draw.circle(screen, (35, 80, 180), (env.x*cell+cell//2, top+env.y*cell+cell//2), 12)
        if env.carrying:
            pygame.draw.circle(screen, (255, 215, 90), (env.x*cell+cell//2, top+env.y*cell+cell//2), 5)
        screen.blit(font.render(f"BFS optimum: {env.optimal_steps} steps, return {env.optimal_return:.2f}", True, (25, 35, 45)), (12, top+env.height*cell+16))
        pygame.display.flip()
        if args.headless and (env.finished or args.method == "manual"):
            path = Path(args.screenshot); path.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(path))
            print(f"Screenshot: {path}; success={info['success']}; steps={env.steps}; return={total:.2f}")
            running = False
        if not args.headless:
            clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
