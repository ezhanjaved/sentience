import jax.numpy as jnp

def obs_decoder(obs):
    decoded_obs = {}

    map = map_decoder(obs)
    tile_info = tiles_decoder(map)
    decoded_obs["map"] = map
    decoded_obs["tiles"] = tile_info

    inventory = inventory_decoder(obs)
    inventory_info = process_inventory(inventory)
    decoded_obs["inventory"] = inventory_info

    print("Info: ", decoded_obs)



def map_decoder(map_obs):
    return map_obs[:8217].reshape(9,11,83)

def inventory_decoder(inv_obs):
    return inv_obs[8217:8233]

def decode_sqrt_values(value):
    return round(float(value * 10) ** 2)

def decode_raw_values(value, df):
    return float(value * df)

def decode_enchantment(value):
    if value == 0:
        return None
    elif value == 1:
        return "fire"
    elif value == 2:
        return "ice"

def decode_item_exists(value):
    if value == 0:
        return "no"
    else:
        return "yes"

def process_inventory(inventory):
    wood = decode_sqrt_values(inventory[0])
    stone = decode_sqrt_values(inventory[1])
    coal = decode_sqrt_values(inventory[2])
    iron = decode_sqrt_values(inventory[3])
    diamond = decode_sqrt_values(inventory[4])
    sapphire = decode_sqrt_values(inventory[5])
    ruby = decode_sqrt_values(inventory[6])
    sapling = decode_sqrt_values(inventory[7])
    torches = decode_sqrt_values(inventory[8])
    arrows = decode_sqrt_values(inventory[9])
    books = decode_raw_values(inventory[10], 2.0)
    pickaxe_level = decode_raw_values(inventory[11], 4.0)
    sword_level = decode_raw_values(inventory[12], 4.0)
    sword_enchantment = decode_enchantment(inventory[13])
    bow_enchantment = decode_enchantment(inventory[14])
    bow = decode_item_exists(inventory[15])

    items = {
        "wood": wood,
        "stone": stone,
        "coal": coal,
        "iron": iron,
        "diamond": diamond,
        "sapphire": sapphire,
        "ruby": ruby,
        "sapling": sapling,
        "torches":torches,
        "arrows":arrows,
        "books":books,
        "pickaxe_level":pickaxe_level,
        "sword_level":sword_level,
        "sword_enchantment":sword_enchantment,
        "bow_enchantment":bow_enchantment,
        "bow":bow
    }

    return items

def tiles_decoder(map):
    tiles = []
    for row in range(9):
        for column in range(11):
            tile = map[row, column]
            one_tile_info = process_one_single_tile(tile)
            tiles.append(one_tile_info)
    return tiles

def process_one_single_tile(tile):
    block_vector = tile[0:37]
    item_vector = tile[37:42]
    mob_vector = tile[42:82].reshape(5,8)

    block_id = int(jnp.argmax(block_vector))
    item_present = bool(jnp.any(item_vector > 0))
    item_id = int(jnp.argmax(item_vector)) if item_present else None

    mobs = []
    for category_id in range(5):
        category = mob_vector[category_id]

        if jnp.any(category > 0):
            mobs.append(
                {
                    "category_id": category_id,
                    "type_id": int(jnp.argmax(category)),
                }
            )

    return {
        "block_id": block_id,
        "item_id": item_id,
        "mobs": mobs,
        "light": float(tile[82]),
    }
