import pygame
import random
import math
import os
import json

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
BLOCK_SIZE = 32
CHUNK_WIDTH = 16
CHUNK_HEIGHT = 64
GRAVITY = 0.8
JUMP_STRENGTH = -12
PLAYER_SPEED = 5
BLOCK_INTERACTION_RANGE = 4  # Maximum blocks away player can interact (in block units)

# Colors
SKY_COLOR = (135, 206, 235)
BLACK = (0, 0, 0)

class PerlinNoise:
    """Simple Perlin noise implementation for terrain generation"""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        self.permutation += self.permutation
    
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, a, b, t):
        return a + t * (b - a)
    
    def grad(self, hash, x):
        return (hash & 1) and x or -x
    
    def noise(self, x):
        X = int(x) & 255
        x -= int(x)
        u = self.fade(x)
        return self.lerp(self.grad(self.permutation[X], x),
                         self.grad(self.permutation[X+1], x-1), u)
    
    def octave_noise(self, x, octaves=4, persistence=0.5):
        value = 0
        amplitude = 1
        frequency = 1
        max_value = 0
        
        for _ in range(octaves):
            value += self.noise(x * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        
        # Normalize to 0-1 range
        normalized = (value / max_value + 1) / 2
        return normalized

class Block:
    def __init__(self, block_type, x, y):
        self.type = block_type
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

class Chunk:
    def __init__(self, chunk_x, noise_generator):
        self.chunk_x = chunk_x
        self.blocks = {}
        self.generate(noise_generator)
    
    def generate(self, noise_generator):
        """Generate blocks for this chunk using Perlin noise with random variations"""
        # Base height for terrain
        base_height = CHUNK_HEIGHT // 2
        
        # Helper function for deterministic randomness based on position
        def deterministic_random(wx, wy, seed_offset=0):
            """Generate deterministic random value between 0-1 based on world coordinates"""
            value = (wx * 73856093) ^ (wy * 19349663) ^ (seed_offset * 83492791)
            return abs((value % 10000) / 10000.0)
        
        # Get previous column's height (for smoothing across chunk boundaries)
        # This ensures smooth transitions between chunks
        prev_height = None
        prev_world_x = self.chunk_x * CHUNK_WIDTH - 1
        # Use noise to calculate previous height (consistent with generation)
        noise1_prev = noise_generator.octave_noise(prev_world_x * 0.05, octaves=3, persistence=0.6)
        noise2_prev = noise_generator.octave_noise(prev_world_x * 0.15, octaves=2, persistence=0.4)
        noise3_prev = noise_generator.octave_noise(prev_world_x * 0.3, octaves=1, persistence=0.3)
        combined_noise_prev = (noise1_prev * 0.8 + noise2_prev * 0.15 + noise3_prev * 0.05)
        height_variation_prev = int(combined_noise_prev * 15)
        prev_height = base_height + height_variation_prev
        prev_height = max(5, min(prev_height, CHUNK_HEIGHT - 8))
        
        for x in range(CHUNK_WIDTH):
            world_x = self.chunk_x * CHUNK_WIDTH + x
            # Generate terrain height using noise with more variation
            # Use multiple noise layers for more randomness
            noise1 = noise_generator.octave_noise(world_x * 0.05, octaves=3, persistence=0.6)
            noise2 = noise_generator.octave_noise(world_x * 0.15, octaves=2, persistence=0.4)
            noise3 = noise_generator.octave_noise(world_x * 0.3, octaves=1, persistence=0.3)
            
            # Combine noises for smoother variation
            # Use more of the smoother noise for gentler terrain
            combined_noise = (noise1 * 0.8 + noise2 * 0.15 + noise3 * 0.05)
            
            # Reduced height variation for smoother terrain
            height_variation = int(combined_noise * 15)  # Reduced from 20 for smoother terrain
            
            height = base_height + height_variation
            
            # SMOOTHING: Limit height difference to maximum 3 blocks from previous column
            if prev_height is not None:
                max_height = prev_height + 3
                min_height = prev_height - 3
                height = max(min_height, min(height, max_height))
            
            # Ensure height is within valid range
            height = max(5, min(height, CHUNK_HEIGHT - 8))
            
            # Store this height for next iteration
            prev_height = height
            
            # Random dirt layer thickness (2-4 blocks) - deterministic per column
            dirt_layers = int(deterministic_random(world_x, 0, 3) * 3) + 2
            
            # Generate blocks from surface all the way to bottom - NO GAPS
            for y in range(CHUNK_HEIGHT):
                world_y = y
                block_x = world_x
                block_y = world_y
                
                if world_y < height:
                    # Air - no block (above surface)
                    continue
                elif world_y == height:
                    # Top surface layer - grass or sand
                    # Use noise to determine if this area should be sand (beach-like areas)
                    sand_noise = noise_generator.octave_noise(world_x * 0.1, octaves=2, persistence=0.5)
                    # Create sand patches - about 15% of surface
                    if sand_noise > 0.65:
                        self.blocks[(x, y)] = Block('sand', block_x, block_y)
                    else:
                        self.blocks[(x, y)] = Block('grass', block_x, block_y)
                elif world_y < height + dirt_layers:
                    # Below grass/sand - dirt or sand layers
                    # Check if surface is sand using the same noise calculation
                    sand_noise = noise_generator.octave_noise(world_x * 0.1, octaves=2, persistence=0.5)
                    is_sand_biome = sand_noise > 0.65
                    
                    if is_sand_biome:
                        # Sand biome - continue with sand
                        if deterministic_random(world_x, world_y, 4) < 0.1:
                            self.blocks[(x, y)] = Block('stone', block_x, block_y)
                        else:
                            self.blocks[(x, y)] = Block('sand', block_x, block_y)
                    else:
                        # Normal biome - dirt layers
                        # 10% chance to place stone instead of dirt for variety
                        if deterministic_random(world_x, world_y, 4) < 0.1:
                            self.blocks[(x, y)] = Block('stone', block_x, block_y)
                        else:
                            self.blocks[(x, y)] = Block('dirt', block_x, block_y)
                else:
                    # Below dirt layers - stone with random caves
                    # Create random caves (air pockets) underground
                    cave_chance = 0.02  # 2% chance per block
                    # More caves deeper underground
                    depth = world_y - (height + dirt_layers)
                    if depth > 10:
                        cave_chance = 0.05  # 5% chance deeper
                    
                    # Use deterministic random for caves
                    if deterministic_random(world_x, world_y, 5) < cave_chance:
                        # Create a small cave (skip this block)
                        continue
                    else:
                        self.blocks[(x, y)] = Block('stone', block_x, block_y)
        
        # Ensure solid floor at the bottom - fill any gaps
        for x in range(CHUNK_WIDTH):
            world_x = self.chunk_x * CHUNK_WIDTH + x
            # Fill bottom 3 layers with stone to prevent falling through
            for floor_y in range(CHUNK_HEIGHT - 3, CHUNK_HEIGHT):
                # Always place stone, even if already exists (ensures no gaps)
                self.blocks[(x, floor_y)] = Block('stone', world_x, floor_y)
        
        # Fill small gaps but preserve caves
        # Only fill single-block gaps to prevent falling through while keeping caves
        for x in range(CHUNK_WIDTH):
            world_x = self.chunk_x * CHUNK_WIDTH + x
            # Find the surface (lowest y with a block)
            surface_y = None
            for y in range(CHUNK_HEIGHT):
                if (x, y) in self.blocks:
                    surface_y = y
                    break
            
            # If we found a surface, ensure continuous blocks to bottom (with small cave exceptions)
            if surface_y is not None:
                last_block_y = surface_y
                for y in range(surface_y + 1, CHUNK_HEIGHT):
                    if (x, y) in self.blocks:
                        # Found a block - check if there was a small gap
                        gap_size = y - last_block_y - 1
                        # Only fill single-block gaps (preserve larger caves)
                        if gap_size == 1:
                            self.blocks[(x, last_block_y + 1)] = Block('stone', world_x, last_block_y + 1)
                        last_block_y = y
                    elif y == CHUNK_HEIGHT - 1:
                        # At bottom - ensure solid floor
                        if (x, y) not in self.blocks:
                            self.blocks[(x, y)] = Block('stone', world_x, y)
            else:
                # No surface found - this shouldn't happen, but ensure we have blocks
                # This is a safety for edge cases
                for y in range(CHUNK_HEIGHT - 10, CHUNK_HEIGHT):
                    if (x, y) not in self.blocks:
                        self.blocks[(x, y)] = Block('stone', world_x, y)
    
    def get_block(self, x, y):
        """Get block at local chunk coordinates"""
        return self.blocks.get((x, y))
    
    def set_block(self, x, y, block_type):
        """Set block at local chunk coordinates"""
        if block_type is None:
            self.blocks.pop((x, y), None)
        else:
            world_x = self.chunk_x * CHUNK_WIDTH + x
            self.blocks[(x, y)] = Block(block_type, world_x, y)

class World:
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 1000000)
        self.noise_generator = PerlinNoise(self.seed)
        self.chunks = {}
        self.block_data = {}
        self.load_block_data()
        self.load_textures()
    
    def load_block_data(self):
        """Load block properties from JSON files"""
        blocks_dir = 'assetdata/blocks'
        if os.path.exists(blocks_dir):
            for filename in os.listdir(blocks_dir):
                if filename.endswith('.json'):
                    block_name = filename[:-5]  # Remove .json extension
                    filepath = os.path.join(blocks_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            self.block_data[block_name] = json.load(f)
                    except Exception as e:
                        print(f"Error loading block data for {block_name}: {e}")
    
    def get_block_property(self, block_type, property_name, default=None):
        """Get a property value for a block type"""
        if block_type in self.block_data:
            return self.block_data[block_type].get(property_name, default)
        return default
    
    def load_textures(self):
        """Load block textures"""
        self.textures = {}
        try:
            self.textures['stone'] = pygame.image.load('assets/blocks/stone.png').convert()
            self.textures['dirt'] = pygame.image.load('assets/blocks/dirt.png').convert()
            self.textures['grass'] = pygame.image.load('assets/blocks/grass.png').convert()
            self.textures['sand'] = pygame.image.load('assets/blocks/sand.png').convert()
            # Try to load cobblestone, fallback to stone if not found
            try:
                self.textures['cobblestone'] = pygame.image.load('assets/blocks/cobblestone.png').convert()
            except:
                self.textures['cobblestone'] = pygame.image.load('assets/blocks/stone.png').convert()
            # Scale textures to block size
            self.textures['stone'] = pygame.transform.scale(self.textures['stone'], (BLOCK_SIZE, BLOCK_SIZE))
            self.textures['dirt'] = pygame.transform.scale(self.textures['dirt'], (BLOCK_SIZE, BLOCK_SIZE))
            self.textures['grass'] = pygame.transform.scale(self.textures['grass'], (BLOCK_SIZE, BLOCK_SIZE))
            self.textures['sand'] = pygame.transform.scale(self.textures['sand'], (BLOCK_SIZE, BLOCK_SIZE))
            self.textures['cobblestone'] = pygame.transform.scale(self.textures['cobblestone'], (BLOCK_SIZE, BLOCK_SIZE))
        except pygame.error as e:
            print(f"Error loading textures: {e}")
            # Create fallback colored blocks
            self.textures['stone'] = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
            self.textures['stone'].fill((128, 128, 128))
            self.textures['dirt'] = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
            self.textures['dirt'].fill((139, 69, 19))
            self.textures['grass'] = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
            self.textures['grass'].fill((34, 139, 34))
            self.textures['sand'] = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
            self.textures['sand'].fill((238, 203, 173))
            self.textures['cobblestone'] = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
            self.textures['cobblestone'].fill((100, 100, 100))
    
    def get_chunk(self, chunk_x):
        """Get or create a chunk"""
        if chunk_x not in self.chunks:
            self.chunks[chunk_x] = Chunk(chunk_x, self.noise_generator)
        return self.chunks[chunk_x]
    
    def get_block(self, world_x, world_y):
        """Get block at world coordinates"""
        # Safety: ensure we don't go below world bounds
        if world_y < 0 or world_y >= CHUNK_HEIGHT:
            return None
        
        chunk_x = world_x // CHUNK_WIDTH
        if world_x < 0:
            chunk_x = (world_x + 1) // CHUNK_WIDTH - 1
        local_x = world_x % CHUNK_WIDTH
        if world_x < 0:
            local_x = CHUNK_WIDTH + (world_x % CHUNK_WIDTH)
        
        chunk = self.get_chunk(chunk_x)
        return chunk.get_block(local_x, world_y)
    
    def set_block(self, world_x, world_y, block_type):
        """Set block at world coordinates"""
        chunk_x = world_x // CHUNK_WIDTH
        if world_x < 0:
            chunk_x = (world_x + 1) // CHUNK_WIDTH - 1
        local_x = world_x % CHUNK_WIDTH
        if world_x < 0:
            local_x = CHUNK_WIDTH + (world_x % CHUNK_WIDTH)
        
        chunk = self.get_chunk(chunk_x)
        chunk.set_block(local_x, world_y, block_type)
    
    def render(self, screen, camera_x, camera_y):
        """Render visible chunks"""
        # Calculate which chunks are visible
        start_chunk = int((camera_x - SCREEN_WIDTH // 2) // (CHUNK_WIDTH * BLOCK_SIZE)) - 1
        end_chunk = int((camera_x + SCREEN_WIDTH // 2) // (CHUNK_WIDTH * BLOCK_SIZE)) + 2
        
        start_y = max(0, int((camera_y - SCREEN_HEIGHT // 2) // BLOCK_SIZE) - 1)
        end_y = min(CHUNK_HEIGHT, int((camera_y + SCREEN_HEIGHT // 2) // BLOCK_SIZE) + 2)
        
        for chunk_x in range(start_chunk, end_chunk + 1):
            chunk = self.get_chunk(chunk_x)
            for (local_x, local_y), block in chunk.blocks.items():
                if start_y <= local_y <= end_y:
                    world_x = chunk.chunk_x * CHUNK_WIDTH + local_x
                    screen_x = world_x * BLOCK_SIZE - camera_x + SCREEN_WIDTH // 2
                    screen_y = local_y * BLOCK_SIZE - camera_y + SCREEN_HEIGHT // 2
                    
                    # Only render if on screen
                    if -BLOCK_SIZE <= screen_x <= SCREEN_WIDTH and -BLOCK_SIZE <= screen_y <= SCREEN_HEIGHT:
                        # Use texture if available, otherwise use fallback
                        texture = self.textures.get(block.type, self.textures.get('stone'))
                        screen.blit(texture, (screen_x, screen_y))

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = BLOCK_SIZE - 4
        self.height = BLOCK_SIZE * 2 - 4
        self.velocity_x = 0
        self.velocity_y = 0
        self.on_ground = False
    
    def update(self, world, keys):
        """Update player position and physics"""
        # Horizontal movement
        self.velocity_x = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.velocity_x = -PLAYER_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.velocity_x = PLAYER_SPEED
        
        # Jump
        if (keys[pygame.K_UP] or keys[pygame.K_SPACE] or keys[pygame.K_w]) and self.on_ground:
            self.velocity_y = JUMP_STRENGTH
            self.on_ground = False
        
        # Apply gravity
        self.velocity_y += GRAVITY
        
        # Simple, reliable collision detection
        # Move horizontally first
        if self.velocity_x != 0:
            new_x = self.x + self.velocity_x
            original_vel_x = self.velocity_x
            # Check if new position has collision
            if not self.check_collision(world, new_x, self.y):
                self.x = new_x
            else:
                # Collision - align to block edge
                self.velocity_x = 0
                if original_vel_x > 0:  # Was moving right
                    block_x = int((new_x + self.width) // BLOCK_SIZE)
                    self.x = block_x * BLOCK_SIZE - self.width
                else:  # Was moving left
                    block_x = int(new_x // BLOCK_SIZE)
                    self.x = (block_x + 1) * BLOCK_SIZE
        
        # Move vertically
        if self.velocity_y != 0:
            new_y = self.y + self.velocity_y
            # Check if new position has collision
            if not self.check_collision(world, self.x, new_y):
                self.y = new_y
                self.on_ground = False
            else:
                # Collision detected
                if self.velocity_y > 0:  # Falling
                    # Land on top of block
                    block_y = int((new_y + self.height) // BLOCK_SIZE)
                    self.y = block_y * BLOCK_SIZE - self.height
                    self.on_ground = True
                    self.velocity_y = 0
                else:  # Hitting ceiling
                    block_y = int(new_y // BLOCK_SIZE)
                    self.y = (block_y + 1) * BLOCK_SIZE
                    self.velocity_y = 0
        
        # CRITICAL: Always ensure player is not inside any blocks
        # This is the most important check to prevent falling through
        while self.check_collision(world, self.x, self.y):
            # Player is inside a block - push out
            # Try moving up
            if not self.check_collision(world, self.x, self.y - 1):
                self.y -= 1
            # Try moving down
            elif not self.check_collision(world, self.x, self.y + 1):
                self.y += 1
            # Try moving left
            elif not self.check_collision(world, self.x - 1, self.y):
                self.x -= 1
            # Try moving right
            elif not self.check_collision(world, self.x + 1, self.y):
                self.x += 1
            else:
                # Can't move, force position
                self.y = int(self.y // BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE
                break
        
        # Check if player should be on ground (only if falling or on ground)
        if self.velocity_y >= 0:  # Only check when falling or stationary
            player_bottom = self.y + self.height
            ground_y = int((player_bottom) // BLOCK_SIZE)
            
            # Check if there's a block directly below player's feet
            has_ground = False
            for check_x in range(int(self.x // BLOCK_SIZE), int((self.x + self.width) // BLOCK_SIZE) + 1):
                if world.get_block(check_x, ground_y) is not None:
                    has_ground = True
                    # Only snap to ground if very close (within 1 pixel) and falling
                    block_top = ground_y * BLOCK_SIZE
                    distance_to_ground = player_bottom - block_top
                    if distance_to_ground >= -0.5 and distance_to_ground <= 1.0:
                        # Very close to ground - snap to it
                        self.y = block_top - self.height
                        self.on_ground = True
                        if self.velocity_y > 0:
                            self.velocity_y = 0
                    elif distance_to_ground < 0.5:
                        # Already on or slightly in ground
                        self.on_ground = True
                    break
            
            if not has_ground:
                self.on_ground = False
    
    def check_collision(self, world, x, y):
        """Check if player collides with any blocks - thorough AABB check"""
        # Calculate which blocks the player overlaps with
        min_block_x = int(x // BLOCK_SIZE)
        max_block_x = int((x + self.width) // BLOCK_SIZE)
        min_block_y = int(y // BLOCK_SIZE)
        max_block_y = int((y + self.height) // BLOCK_SIZE)
        
        # Check all blocks the player rectangle overlaps
        for block_x in range(min_block_x, max_block_x + 1):
            for block_y in range(min_block_y, max_block_y + 1):
                block = world.get_block(block_x, block_y)
                if block is not None:
                    # AABB collision detection
                    block_world_x = block_x * BLOCK_SIZE
                    block_world_y = block_y * BLOCK_SIZE
                    # Check if rectangles overlap
                    if (x < block_world_x + BLOCK_SIZE and 
                        x + self.width > block_world_x and
                        y < block_world_y + BLOCK_SIZE and 
                        y + self.height > block_world_y):
                        return True
        return False
    
    def get_collision_info(self, world, x, y):
        """Get detailed collision information"""
        collisions = []
        min_block_x = int(x // BLOCK_SIZE)
        max_block_x = int((x + self.width) // BLOCK_SIZE)
        min_block_y = int(y // BLOCK_SIZE)
        max_block_y = int((y + self.height) // BLOCK_SIZE)
        
        for block_x in range(min_block_x - 1, max_block_x + 2):
            for block_y in range(min_block_y - 1, max_block_y + 2):
                block = world.get_block(block_x, block_y)
                if block is not None:
                    block_world_x = block_x * BLOCK_SIZE
                    block_world_y = block_y * BLOCK_SIZE
                    if (x < block_world_x + BLOCK_SIZE and x + self.width > block_world_x and
                        y < block_world_y + BLOCK_SIZE and y + self.height > block_world_y):
                        collisions.append((block_x, block_y, block_world_x, block_world_y))
        return collisions
    
    def render(self, screen, camera_x, camera_y):
        """Render player"""
        screen_x = self.x - camera_x + SCREEN_WIDTH // 2
        screen_y = self.y - camera_y + SCREEN_HEIGHT // 2
        pygame.draw.rect(screen, (255, 0, 0), (screen_x, screen_y, self.width, self.height))

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        self.selected_option = 0  # 0 = Singleplayer
        self.in_menu = True
        
        # Create pixelated fonts by using small size and scaling up
        # This creates a more pixelated/retro look
        base_title_size = 36
        base_menu_size = 24
        
        # Create fonts without antialiasing for pixelated look
        self.title_font = pygame.font.Font(None, base_title_size)
        self.menu_font = pygame.font.Font(None, base_menu_size)
        
        # Store scale factors
        self.title_scale = 2
        self.menu_scale = 2
    
    def handle_events(self):
        """Handle menu events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.in_menu = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.in_menu = False
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    self.selected_option = 0
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    self.selected_option = 0  # Only one option, but keep for consistency
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    if self.selected_option == 0:  # Singleplayer
                        self.in_menu = False
    
    def render(self):
        """Render the main menu"""
        # Draw background gradient
        for y in range(SCREEN_HEIGHT):
            color_value = int(50 + (y / SCREEN_HEIGHT) * 50)
            pygame.draw.line(self.screen, (color_value, color_value, color_value + 20), 
                           (0, y), (SCREEN_WIDTH, y))
        
        # Draw game title "IDKblocks" with pixelated scaling
        title_surface = self.title_font.render("IDKblocks", False, (255, 255, 255))  # False = no antialiasing
        title_scaled = pygame.transform.scale(title_surface, 
                                              (title_surface.get_width() * self.title_scale, 
                                               title_surface.get_height() * self.title_scale))
        title_rect = title_scaled.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 3))
        
        # Draw title shadow for depth
        shadow_surface = self.title_font.render("IDKblocks", False, (0, 0, 0))
        shadow_scaled = pygame.transform.scale(shadow_surface,
                                              (shadow_surface.get_width() * self.title_scale,
                                               shadow_surface.get_height() * self.title_scale))
        shadow_rect = shadow_scaled.get_rect(center=(SCREEN_WIDTH // 2 + 4, SCREEN_HEIGHT // 3 + 4))
        self.screen.blit(shadow_scaled, shadow_rect)
        self.screen.blit(title_scaled, title_rect)
        
        # Draw menu options
        menu_y = SCREEN_HEIGHT // 2 + 50
        options = ["Singleplayer"]
        
        for i, option in enumerate(options):
            # Render with pixelated scaling
            if i == self.selected_option:
                # Highlighted option
                option_surface = self.menu_font.render(f"> {option} <", False, (255, 255, 0))
            else:
                option_surface = self.menu_font.render(option, False, (200, 200, 200))
            
            option_scaled = pygame.transform.scale(option_surface,
                                                  (option_surface.get_width() * self.menu_scale,
                                                   option_surface.get_height() * self.menu_scale))
            option_rect = option_scaled.get_rect(center=(SCREEN_WIDTH // 2, menu_y + i * 80))
            
            # Draw shadow
            if i == self.selected_option:
                shadow_surface = self.menu_font.render(f"> {option} <", False, (0, 0, 0))
            else:
                shadow_surface = self.menu_font.render(option, False, (0, 0, 0))
            shadow_scaled = pygame.transform.scale(shadow_surface,
                                                   (shadow_surface.get_width() * self.menu_scale,
                                                    shadow_surface.get_height() * self.menu_scale))
            shadow_rect = shadow_scaled.get_rect(center=(SCREEN_WIDTH // 2 + 2, menu_y + i * 80 + 2))
            self.screen.blit(shadow_scaled, shadow_rect)
            self.screen.blit(option_scaled, option_rect)
        
        pygame.display.flip()
    
    def run(self):
        """Run the main menu loop"""
        while self.in_menu and self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        return self.running

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("IDKblocks")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create world
        self.world = World()
        
        # Breaking state
        # Breaking state
        self.breaking_block = None  # (world_x, world_y) of block being broken
        self.breaking_start_time = 0
        self.breaking_progress = 0.0  # 0.0 to 1.0
        self.mouse_held = False
        self.breaking_overlays = []
        self.breaking_particles = []  # Particles for breaking effects
        self.create_breaking_overlays()
        
        # Inventory system
        self.inventory = {'dirt': 10, 'grass': 10, 'stone': 5, 'cobblestone': 5, 'sand': 5}  # Starting items
        self.selected_slot = 0  # 0-8 for hotbar
        self.hotbar_slots = ['dirt', 'grass', 'stone', 'cobblestone', 'sand', None, None, None, None]
    
    def create_breaking_overlays(self):
        """Create crack overlay textures for breaking animation"""
        self.breaking_overlays = []
        # Create 10 stages of breaking (0% to 100%)
        for stage in range(10):
            overlay = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            # Draw crack pattern based on stage
            crack_intensity = stage / 9.0
            num_cracks = int(crack_intensity * 8) + 1
            
            for i in range(num_cracks):
                # Random crack positions
                start_x = random.randint(0, BLOCK_SIZE)
                start_y = random.randint(0, BLOCK_SIZE)
                end_x = random.randint(0, BLOCK_SIZE)
                end_y = random.randint(0, BLOCK_SIZE)
                
                # Draw crack line with varying opacity
                alpha = int(200 * crack_intensity)
                color = (255, 255, 255, alpha)
                pygame.draw.line(overlay, color, (start_x, start_y), (end_x, end_y), 2)
            
            self.breaking_overlays.append(overlay)
        
        # Pre-generate chunks around spawn to prevent phantom chunks
        # Generate chunks from -2 to +2 to ensure smooth terrain
        spawn_chunk_x = 0
        for chunk_offset in range(-2, 3):
            chunk_x = spawn_chunk_x + chunk_offset
            self.world.get_chunk(chunk_x)  # This will generate the chunk if it doesn't exist
        
        # Create player at surface level
        spawn_x = 0
        spawn_y = 0
        # Find surface at spawn - look from top to bottom
        for y in range(CHUNK_HEIGHT):
            if self.world.get_block(spawn_x, y) is not None:
                spawn_y = y * BLOCK_SIZE - BLOCK_SIZE * 2
                break
        # If no surface found, spawn at a safe height
        if spawn_y == 0:
            spawn_y = (CHUNK_HEIGHT // 2) * BLOCK_SIZE
        
        self.player = Player(spawn_x * BLOCK_SIZE, spawn_y)
        self.camera_x = self.player.x
        self.camera_y = self.player.y
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                # Hotbar selection with number keys 1-9
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    slot = event.key - pygame.K_1
                    if 0 <= slot < len(self.hotbar_slots):
                        self.selected_slot = slot
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_held = True
                    # Start breaking block
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_x = int((mouse_x - SCREEN_WIDTH // 2 + self.camera_x) // BLOCK_SIZE)
                    world_y = int((mouse_y - SCREEN_HEIGHT // 2 + self.camera_y) // BLOCK_SIZE)
                    
                    # Calculate distance from player to block
                    player_block_x = int(self.player.x // BLOCK_SIZE)
                    player_block_y = int(self.player.y // BLOCK_SIZE)
                    distance_x = abs(world_x - player_block_x)
                    distance_y = abs(world_y - player_block_y)
                    distance = max(distance_x, distance_y)
                    
                    # Only interact if within range and block exists
                    if distance <= BLOCK_INTERACTION_RANGE:
                        block = self.world.get_block(world_x, world_y)
                        if block is not None:
                            self.breaking_block = (world_x, world_y)
                            self.breaking_start_time = pygame.time.get_ticks()
                            self.breaking_progress = 0.0
                elif event.button == 3:  # Right click - place block
                    # Block breaking/placing with range check
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_x = int((mouse_x - SCREEN_WIDTH // 2 + self.camera_x) // BLOCK_SIZE)
                    world_y = int((mouse_y - SCREEN_HEIGHT // 2 + self.camera_y) // BLOCK_SIZE)
                    
                    # Calculate distance from player to block
                    player_block_x = int(self.player.x // BLOCK_SIZE)
                    player_block_y = int(self.player.y // BLOCK_SIZE)
                    distance_x = abs(world_x - player_block_x)
                    distance_y = abs(world_y - player_block_y)
                    distance = max(distance_x, distance_y)  # Chebyshev distance (allows diagonal)
                    
                    # Only interact if within range
                    if distance <= BLOCK_INTERACTION_RANGE:
                        if self.world.get_block(world_x, world_y) is None:
                            # Check if block would be inside player
                            block_world_x = world_x * BLOCK_SIZE
                            block_world_y = world_y * BLOCK_SIZE
                            
                            # Check if block overlaps with player's collision box
                            player_left = self.player.x
                            player_right = self.player.x + self.player.width
                            player_top = self.player.y
                            player_bottom = self.player.y + self.player.height
                            
                            block_left = block_world_x
                            block_right = block_world_x + BLOCK_SIZE
                            block_top = block_world_y
                            block_bottom = block_world_y + BLOCK_SIZE
                            
                            # Check for overlap
                            overlaps = not (block_right <= player_left or 
                                          block_left >= player_right or 
                                          block_bottom <= player_top or 
                                          block_top >= player_bottom)
                            
                            # Only place if not overlapping with player
                            if not overlaps:
                                # Get selected block from hotbar
                                selected_block = self.hotbar_slots[self.selected_slot]
                                if selected_block and self.inventory.get(selected_block, 0) > 0:
                                    self.world.set_block(world_x, world_y, selected_block)
                                    self.inventory[selected_block] -= 1
                                    # Check if there's grass below this block and convert it to dirt
                                    block_below = self.world.get_block(world_x, world_y + 1)
                                    if block_below is not None and block_below.type == 'grass':
                                        self.world.set_block(world_x, world_y + 1, 'dirt')
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button released
                    self.mouse_held = False
                    self.breaking_block = None
                    self.breaking_progress = 0.0
    
    def update_breaking(self):
        """Update block breaking progress"""
        if self.breaking_block is None or not self.mouse_held:
            return
        
        world_x, world_y = self.breaking_block
        block = self.world.get_block(world_x, world_y)
        
        # Check if block still exists and is in range
        if block is None:
            self.breaking_block = None
            self.breaking_progress = 0.0
            return
        
        player_block_x = int(self.player.x // BLOCK_SIZE)
        player_block_y = int(self.player.y // BLOCK_SIZE)
        distance_x = abs(world_x - player_block_x)
        distance_y = abs(world_y - player_block_y)
        distance = max(distance_x, distance_y)
        
        if distance > BLOCK_INTERACTION_RANGE:
            self.breaking_block = None
            self.breaking_progress = 0.0
            return
        
        # Get breaking time from block data
        breaking_time = self.world.get_block_property(block.type, 'breaking_time', 1.0)
        breaking_time_ms = breaking_time * 1000  # Convert to milliseconds
        
        # Calculate progress
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.breaking_start_time
        self.breaking_progress = min(elapsed / breaking_time_ms, 1.0)
        
        # Check if block is broken
        if self.breaking_progress >= 1.0:
            # Get drops from block data
            drops = self.world.get_block_property(block.type, 'drops', [])
            
            # Break the block
            self.world.set_block(world_x, world_y, None)
            
            # Add drops to inventory
            for drop in drops:
                item = drop.get('item', block.type)
                count = drop.get('count', 1)
                if item in self.inventory:
                    self.inventory[item] += count
            
            # Create breaking effect particles
            self.create_breaking_effect(world_x, world_y, block.type)
            
            # Reset breaking state
            self.breaking_block = None
            self.breaking_progress = 0.0
    
    def create_breaking_effect(self, world_x, world_y, block_type):
        """Create particle effects when a block breaks"""
        block_screen_x = world_x * BLOCK_SIZE
        block_screen_y = world_y * BLOCK_SIZE
        
        # Create 8-12 particles
        num_particles = random.randint(8, 12)
        for _ in range(num_particles):
            particle = {
                'x': block_screen_x + BLOCK_SIZE // 2,
                'y': block_screen_y + BLOCK_SIZE // 2,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-5, -1),
                'life': 30,  # frames
                'size': random.randint(2, 4),
                'color': self.get_block_particle_color(block_type)
            }
            self.breaking_particles.append(particle)
    
    def get_block_particle_color(self, block_type):
        """Get particle color based on block type"""
        colors = {
            'dirt': (139, 69, 19),
            'grass': (34, 139, 34),
            'stone': (128, 128, 128),
            'cobblestone': (100, 100, 100),
            'sand': (238, 203, 173)
        }
        return colors.get(block_type, (128, 128, 128))
    
    def update_particles(self):
        """Update and remove expired particles"""
        for particle in self.breaking_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                self.breaking_particles.remove(particle)
    
    def render_breaking_animation(self, screen):
        """Render breaking animation overlay"""
        if self.breaking_block is None:
            return
        
        world_x, world_y = self.breaking_block
        block_screen_x = world_x * BLOCK_SIZE - self.camera_x + SCREEN_WIDTH // 2
        block_screen_y = world_y * BLOCK_SIZE - self.camera_y + SCREEN_HEIGHT // 2
        
        # Get overlay stage (0-9)
        overlay_stage = int(self.breaking_progress * 9)
        overlay_stage = min(overlay_stage, 9)
        
        if overlay_stage < len(self.breaking_overlays):
            screen.blit(self.breaking_overlays[overlay_stage], (block_screen_x, block_screen_y))
    
    def render_particles(self, screen):
        """Render breaking effect particles"""
        for particle in self.breaking_particles:
            # Convert world coordinates to screen coordinates
            screen_x = particle['x'] - self.camera_x + SCREEN_WIDTH // 2
            screen_y = particle['y'] - self.camera_y + SCREEN_HEIGHT // 2
            
            # Only render if on screen
            if -10 <= screen_x <= SCREEN_WIDTH + 10 and -10 <= screen_y <= SCREEN_HEIGHT + 10:
                # Calculate alpha based on life
                alpha = int(255 * (particle['life'] / 30.0))
                color = particle['color']
                
                # Draw particle
                pygame.draw.circle(screen, color, (int(screen_x), int(screen_y)), particle['size'])
    
    def update(self):
        """Update game state"""
        keys = pygame.key.get_pressed()
        self.player.update(self.world, keys)
        
        # Update breaking progress
        self.update_breaking()
        
        # Update particles
        self.update_particles()
        
        # Smooth camera follow
        self.camera_x += (self.player.x - self.camera_x) * 0.1
        self.camera_y += (self.player.y - self.camera_y) * 0.1
    
    def render(self):
        """Render everything"""
        # Draw sky
        self.screen.fill(SKY_COLOR)
        
        # Render world
        self.world.render(self.screen, self.camera_x, self.camera_y)
        
        # Render breaking animation
        self.render_breaking_animation(self.screen)
        
        # Render particles
        self.render_particles(self.screen)
        
        # Render player
        self.player.render(self.screen, self.camera_x, self.camera_y)
        
        # Draw selected block outline
        mouse_x, mouse_y = pygame.mouse.get_pos()
        world_x = int((mouse_x - SCREEN_WIDTH // 2 + self.camera_x) // BLOCK_SIZE)
        world_y = int((mouse_y - SCREEN_HEIGHT // 2 + self.camera_y) // BLOCK_SIZE)
        
        # Calculate distance from player to block
        player_block_x = int(self.player.x // BLOCK_SIZE)
        player_block_y = int(self.player.y // BLOCK_SIZE)
        distance_x = abs(world_x - player_block_x)
        distance_y = abs(world_y - player_block_y)
        distance = max(distance_x, distance_y)
        
        # Only show outline if within range
        if distance <= BLOCK_INTERACTION_RANGE:
            # Calculate screen position of selected block
            block_screen_x = world_x * BLOCK_SIZE - self.camera_x + SCREEN_WIDTH // 2
            block_screen_y = world_y * BLOCK_SIZE - self.camera_y + SCREEN_HEIGHT // 2
            
            # Draw outline (white rectangle)
            outline_rect = pygame.Rect(block_screen_x, block_screen_y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, (255, 255, 255), outline_rect, 2)
        
        # Draw inventory hotbar
        self.render_hotbar()
        
        pygame.display.flip()
    
    def render_hotbar(self):
        """Render the inventory hotbar at the top of the screen"""
        hotbar_height = 60
        slot_size = 50
        slot_spacing = 5
        hotbar_width = len(self.hotbar_slots) * (slot_size + slot_spacing) - slot_spacing
        hotbar_x = (SCREEN_WIDTH - hotbar_width) // 2
        hotbar_y = 10
        
        # Draw hotbar background
        hotbar_bg = pygame.Surface((hotbar_width + 20, hotbar_height + 20))
        hotbar_bg.fill((50, 50, 50))
        hotbar_bg.set_alpha(200)
        self.screen.blit(hotbar_bg, (hotbar_x - 10, hotbar_y - 10))
        
        # Draw slots
        font = pygame.font.Font(None, 24)
        for i, block_type in enumerate(self.hotbar_slots):
            slot_x = hotbar_x + i * (slot_size + slot_spacing)
            slot_y = hotbar_y
            
            # Draw slot background
            if i == self.selected_slot:
                # Selected slot - highlight
                pygame.draw.rect(self.screen, (255, 255, 255), 
                              (slot_x - 2, slot_y - 2, slot_size + 4, slot_size + 4), 3)
                slot_bg_color = (100, 100, 100)
            else:
                slot_bg_color = (70, 70, 70)
            
            pygame.draw.rect(self.screen, slot_bg_color, (slot_x, slot_y, slot_size, slot_size))
            pygame.draw.rect(self.screen, (30, 30, 30), (slot_x, slot_y, slot_size, slot_size), 2)
            
            # Draw block icon if slot has a block type
            if block_type:
                count = self.inventory.get(block_type, 0)
                if count > 0 and block_type in self.world.textures:
                    # Scale texture to fit slot
                    icon = pygame.transform.scale(self.world.textures[block_type], 
                                                (slot_size - 8, slot_size - 8))
                    self.screen.blit(icon, (slot_x + 4, slot_y + 4))
                    
                    # Draw count
                    count_text = font.render(str(count), True, (255, 255, 255))
                    count_rect = count_text.get_rect(bottomright=(slot_x + slot_size - 4, slot_y + slot_size - 4))
                    # Draw shadow for text
                    shadow_text = font.render(str(count), True, (0, 0, 0))
                    self.screen.blit(shadow_text, (count_rect.x + 1, count_rect.y + 1))
                    self.screen.blit(count_text, count_rect)
                else:
                    # Empty slot - draw gray overlay
                    overlay = pygame.Surface((slot_size, slot_size))
                    overlay.fill((0, 0, 0))
                    overlay.set_alpha(150)
                    self.screen.blit(overlay, (slot_x, slot_y))
            
            # Draw slot number (1-9)
            num_text = font.render(str(i + 1), True, (200, 200, 200))
            num_rect = num_text.get_rect(topleft=(slot_x + 2, slot_y + 2))
            self.screen.blit(num_text, num_rect)
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    menu = MainMenu(screen)
    if menu.run():  # If menu returns True (not quit), start game
        game = Game()
        game.run()
    pygame.quit()

